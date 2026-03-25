"""Modal app for remote steering inference on GPU.

Loads the model once on a single container, then runs the entire sweep
with batched generation to maximize GPU utilization.
"""

import logging

import modal

log = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3.5-9B"


def download_model() -> None:
    """Pre-download model weights into the image cache at build time."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto")
    AutoTokenizer.from_pretrained(MODEL_NAME)


# Heavy layers first (cached across run.py edits), local code last.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "accelerate", "numpy", "scikit-learn",
        "fire", "python-dotenv", "pydantic",
    )
    .run_function(download_model)
    .add_local_file("run.py", "/root/run.py")
)

app = modal.App("autosteer", image=image)


@app.cls(gpu="A10G", timeout=14400)
class SteeringInference:
    """Single-container inference — loads model once, runs full sweep batched."""

    @modal.enter()
    def setup(self) -> None:
        """Load model into GPU memory on container start."""
        import sys

        import torch

        sys.path.insert(0, "/root")
        from run import load_model

        self.bundle = load_model(MODEL_NAME)
        log.info("Model device: %s, CUDA available: %s", self.bundle.device, torch.cuda.is_available())
        if torch.cuda.is_available():
            log.info("GPU: %s, VRAM: %.1f GB", torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory / 1e9)

    @modal.method()
    def run_sweep(
        self,
        eval_prompts: list[str],
        vector_bytes_by_layer: dict[int, bytes],
        alpha_values: list[float],
        seed: int,
    ) -> dict:
        """Run full baseline + steered sweep on a single GPU with batched generation.

        Returns dict with "baselines" and "steered" keys matching run.py format.
        """
        import time

        from run import (
            deserialize_vector,
            generate_baseline_batch,
            generate_with_steering_batch,
        )

        # Baselines — one batched call for all prompts
        log.info("Generating %d baselines (batched)...", len(eval_prompts))
        t0 = time.monotonic()
        baselines = generate_baseline_batch(self.bundle, eval_prompts, seed=seed)
        log.info("Baselines done in %.1fs", time.monotonic() - t0)

        # Steered — one batched call per (layer, alpha) pair
        layers = sorted(vector_bytes_by_layer.keys())
        n_combos = len(layers) * len(alpha_values)
        total = n_combos * len(eval_prompts)
        log.info(
            "Running %d steered generations (%d layers x %d alphas x %d prompts, batched per combo)...",
            total, len(layers), len(alpha_values), len(eval_prompts),
        )

        steered: list[dict] = []
        done = 0
        t0 = time.monotonic()
        for layer_idx in layers:
            vector = deserialize_vector(vector_bytes_by_layer[layer_idx])
            for alpha in alpha_values:
                outputs = generate_with_steering_batch(
                    self.bundle, eval_prompts, vector, alpha, seed=seed,
                )
                steered.append({"layer": layer_idx, "alpha": alpha, "outputs": outputs})
                done += 1
                elapsed = time.monotonic() - t0
                log.info(
                    "  %d/%d combos done (%.1fs elapsed, %.1fs/combo, ~%.1fs/prompt)",
                    done, n_combos, elapsed, elapsed / done, elapsed / (done * len(eval_prompts)),
                )

        elapsed = time.monotonic() - t0
        log.info(
            "Sweep complete: %d combos (%d prompts) in %.1fs (%.1fs/prompt)",
            n_combos, total, elapsed, elapsed / max(total, 1),
        )
        return {"baselines": baselines, "steered": steered}
