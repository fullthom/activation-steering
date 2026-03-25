"""Autosteer: change LLM behavior using activation steering vectors derived from natural language."""

import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import dotenv
import fire
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContrastivePair:
    """A prompt with positive (exhibits behavior) and negative (doesn't) completions."""

    prompt: str
    positive: str
    negative: str


@dataclass(frozen=True)
class SteeringVector:
    """A unit-normalized direction vector for steering at a specific layer."""

    vector: torch.Tensor
    layer: int
    behavior: str
    magnitude: float = 0.0  # Pre-normalization magnitude for cross-layer comparison
    explained_variances: tuple[float, ...] = ()  # PCA explained variance ratios for top components (0-1 each)
    probe_accuracy: float = 0.0  # Logistic regression cross-val accuracy (0-1)
    consistency: float = 0.0  # Mean pairwise cosine sim of per-pair diff vectors (0-1)
    stability: float = 0.0  # Split-half stability: mean cosine sim between random half-split vectors
    snr: float = 0.0  # Signal-to-noise ratio: mean diff norm / std of diff norms
    norm_sensitivity: float = 0.0  # Cosine sim between raw-mean and normalized-mean vectors (1.0 = no distortion)


@dataclass(frozen=True)
class ModelBundle:
    """A loaded model with its tokenizer and config info."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    n_layers: int
    d_model: int
    device: torch.device


# ── Storage ──────────────────────────────────────────────────────────────────

BEHAVIORS_DIR = Path("behaviors")


def _behavior_dir(behavior: str) -> Path:
    """Return the directory for a behavior, creating it if needed."""
    slug = re.sub(r"[^a-z0-9]+", "_", behavior.lower()).strip("_")
    path = BEHAVIORS_DIR / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_slug(model_name: str) -> str:
    """Turn a model name like 'Qwen/Qwen3.5-9B' into 'qwen3.5-9b'."""
    return re.sub(r"[^a-z0-9._-]+", "-", model_name.split("/")[-1].lower()).strip("-")


def _load_pairs(path: Path) -> list[ContrastivePair]:
    """Load contrastive pairs from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"No pairs file found at {path}. Create a pairs.json with contrastive examples.")
    data = json.loads(path.read_text())
    return [ContrastivePair(**d) for d in data]


def _save_vector(vector: SteeringVector, path: Path) -> None:
    """Save steering vector to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vector": vector.vector, "layer": vector.layer, "behavior": vector.behavior,
            "magnitude": vector.magnitude, "explained_variances": list(vector.explained_variances),
            "probe_accuracy": vector.probe_accuracy, "consistency": vector.consistency,
            "stability": vector.stability, "snr": vector.snr,
            "norm_sensitivity": vector.norm_sensitivity,
        },
        path,
    )


def _load_vector(path: Path) -> SteeringVector | None:
    """Load steering vector from disk, or None if not cached."""
    if not path.exists():
        return None
    data = torch.load(path, weights_only=True)
    magnitude = float(data.pop("magnitude", 0.0))
    raw_variances = data.pop("explained_variances", [])
    # Migrate old format: single explained_variance + pc2/pc3/pc4plus fields
    if not raw_variances:
        ev = float(data.pop("explained_variance", 0.0))
        pc2 = float(data.pop("pc2_explained", 0.0))
        pc3 = float(data.pop("pc3_explained", 0.0))
        pc4plus = float(data.pop("pc4plus_explained", 0.0))
        raw_variances = [ev, pc2, pc3, pc4plus] if ev > 0 else []
    else:
        # Clean up any old keys that might co-exist
        data.pop("explained_variance", None)
        data.pop("pc2_explained", None)
        data.pop("pc3_explained", None)
        data.pop("pc4plus_explained", None)
    explained_variances = tuple(float(v) for v in raw_variances)
    probe_accuracy = float(data.pop("probe_accuracy", 0.0))
    consistency = float(data.pop("consistency", 0.0))
    stability = float(data.pop("stability", 0.0))
    snr = float(data.pop("snr", 0.0))
    norm_sensitivity = float(data.pop("norm_sensitivity", 0.0))
    return SteeringVector(
        **data, magnitude=magnitude, explained_variances=explained_variances,
        probe_accuracy=probe_accuracy, consistency=consistency, stability=stability,
        snr=snr, norm_sensitivity=norm_sensitivity,
    )


def serialize_vector(vector: SteeringVector) -> bytes:
    """Serialize a SteeringVector to bytes for remote transfer."""
    buf = io.BytesIO()
    torch.save(
        {
            "vector": vector.vector, "layer": vector.layer, "behavior": vector.behavior,
            "magnitude": vector.magnitude, "explained_variances": list(vector.explained_variances),
            "probe_accuracy": vector.probe_accuracy, "consistency": vector.consistency,
            "stability": vector.stability, "snr": vector.snr,
            "norm_sensitivity": vector.norm_sensitivity,
        },
        buf,
    )
    return buf.getvalue()


def deserialize_vector(data: bytes) -> SteeringVector:
    """Deserialize a SteeringVector from bytes."""
    buf = io.BytesIO(data)
    d = torch.load(buf, weights_only=True)
    return SteeringVector(
        vector=d["vector"],
        layer=int(d["layer"]),
        behavior=str(d["behavior"]),
        magnitude=float(d["magnitude"]),
        explained_variances=tuple(float(v) for v in d["explained_variances"]),
        probe_accuracy=float(d["probe_accuracy"]),
        consistency=float(d["consistency"]),
        stability=float(d["stability"]),
        snr=float(d["snr"]),
        norm_sensitivity=float(d["norm_sensitivity"]),
    )


# ── Prompt Constants ─────────────────────────────────────────────────────────


# Separate eval prompts the model has never seen during vector computation
EVAL_PROMPTS: list[str] = [
    "How do black holes form?",
    "What makes the sky blue?",
    "Explain how a microwave oven heats food.",
    "What is the greenhouse effect?",
    "How does memory work in the human brain?",
    "What causes volcanoes to erupt?",
    "Explain how computers store data.",
    "What is dark matter?",
    "How do antibiotics work?",
    "What causes rainbows?",
]

# ── Stage 2: Model Loading ───────────────────────────────────────────────────


def _get_device() -> torch.device:
    """Detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str = "Qwen/Qwen3.5-9B") -> ModelBundle:
    """Load a HuggingFace model and tokenizer."""
    device = _get_device()
    log.info("Loading %s on %s", model_name, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use "auto" on CUDA (accelerate places layers optimally), device object elsewhere
    device_map: str | torch.device = "auto" if device.type == "cuda" else device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()

    # Handle nested configs (e.g. Qwen3.5 uses text_config)
    config = model.config
    text_config = getattr(config, "text_config", config)
    n_layers = text_config.num_hidden_layers
    d_model = text_config.hidden_size

    log.info("Loaded: %d layers, d_model=%d", n_layers, d_model)
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        n_layers=n_layers,
        d_model=d_model,
        device=device,
    )


def _format_chat(tokenizer: PreTrainedTokenizerBase, user_message: str) -> str:
    """Format a user message using the model's chat template if available."""
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": user_message}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    return user_message


# ── Stage 3: Activation Extraction ──────────────────────────────────────────


def _get_layer_module(model: PreTrainedModel, layer: int) -> torch.nn.Module:
    """Get the transformer layer module by index. Works across architectures."""
    # Most HF models use model.model.layers or model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer]
    raise ValueError(f"Cannot find layer modules in {type(model).__name__}")


def _get_last_token_activation(
    bundle: ModelBundle,
    text: str,
    layer: int,
) -> torch.Tensor:
    """Run forward pass and return residual stream activation at last token for given layer."""
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> None:  # noqa: A002
        # Layer output is typically a tuple; first element is the hidden state
        hidden = output[0] if isinstance(output, tuple) else output
        captured["activation"] = hidden[0, -1, :].detach().cpu()

    layer_module = _get_layer_module(bundle.model, layer)
    handle = layer_module.register_forward_hook(hook_fn)

    try:
        inputs = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)
        with torch.no_grad():
            bundle.model(**inputs)
    finally:
        handle.remove()

    return captured["activation"].float()


def extract_activations(
    bundle: ModelBundle,
    pairs: list[ContrastivePair],
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract last-token residual stream activations for positive and negative completions.

    Returns (pos_activations, neg_activations), each shape [n_pairs, d_model].
    """
    pos_acts: list[torch.Tensor] = []
    neg_acts: list[torch.Tensor] = []

    for i, pair in enumerate(pairs):
        pos_text = _format_chat(bundle.tokenizer, pair.prompt + " " + pair.positive)
        neg_text = _format_chat(bundle.tokenizer, pair.prompt + " " + pair.negative)

        pos_acts.append(_get_last_token_activation(bundle, pos_text, layer))
        neg_acts.append(_get_last_token_activation(bundle, neg_text, layer))

        log.info("Extracted activations: %d/%d", i + 1, len(pairs))

    log.info("Done extracting %d pairs at layer %d", len(pairs), layer)
    return torch.stack(pos_acts), torch.stack(neg_acts)


def _get_last_token_activations_all_layers(
    bundle: ModelBundle,
    text: str,
) -> torch.Tensor:
    """Run one forward pass and return last-token activations for ALL layers.

    Returns: Tensor of shape [n_layers, d_model].
    """
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):  # noqa: ANN202
        def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> None:  # noqa: A002
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden[0, -1, :].detach().cpu()
        return hook_fn

    handles = []
    for i in range(bundle.n_layers):
        layer_module = _get_layer_module(bundle.model, i)
        handles.append(layer_module.register_forward_hook(make_hook(i)))

    try:
        inputs = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)
        with torch.no_grad():
            bundle.model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return torch.stack([captured[i].float() for i in range(bundle.n_layers)])


def extract_activations_all_layers(
    bundle: ModelBundle,
    pairs: list[ContrastivePair],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract last-token activations for ALL layers in one forward pass per pair.

    Returns (pos_activations, neg_activations), each shape [n_pairs, n_layers, d_model].
    """
    pos_acts: list[torch.Tensor] = []
    neg_acts: list[torch.Tensor] = []

    for i, pair in enumerate(pairs):
        pos_text = _format_chat(bundle.tokenizer, pair.prompt + " " + pair.positive)
        neg_text = _format_chat(bundle.tokenizer, pair.prompt + " " + pair.negative)

        pos_acts.append(_get_last_token_activations_all_layers(bundle, pos_text))
        neg_acts.append(_get_last_token_activations_all_layers(bundle, neg_text))

        log.info("Extracted all-layer activations: %d/%d", i + 1, len(pairs))

    log.info("Done extracting %d pairs across %d layers", len(pairs), bundle.n_layers)
    return torch.stack(pos_acts), torch.stack(neg_acts)


def probe_layer(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_components: int = 50,
) -> float:
    """Train a logistic regression probe to measure linear separability at a layer.

    pos_activations: [n, d_model], neg_activations: [n, d_model]
    PCA reduces to n_components dims first to prevent overfitting in high-d spaces.
    Returns: mean cross-validated accuracy (0-1).
    """
    x = torch.cat([pos_activations, neg_activations]).numpy()
    y = np.array([1] * len(pos_activations) + [0] * len(neg_activations))

    # PCA to prevent memorization when d_model >> n_samples
    n_comp = min(n_components, x.shape[0] - 1, x.shape[1])
    x = PCA(n_components=n_comp).fit_transform(x)

    clf = LogisticRegression(max_iter=1000, C=0.01)
    scores = cross_val_score(clf, x, y, cv=min(5, len(pos_activations)))
    return float(scores.mean())


# ── Stage 4: Steering Vector Computation ─────────────────────────────────────


def compute_consistency(pos_activations: torch.Tensor, neg_activations: torch.Tensor) -> float:
    """Mean pairwise cosine similarity of per-pair difference vectors.

    pos_activations: [n_pairs, d_model]
    neg_activations: [n_pairs, d_model]
    Returns: scalar in [-1, 1], higher means pairs agree on direction.
    """
    diffs = pos_activations - neg_activations
    norms = diffs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    diffs_normed = diffs / norms
    pairwise = diffs_normed @ diffs_normed.t()
    mask = ~torch.eye(len(diffs), dtype=torch.bool, device=pairwise.device)
    return pairwise[mask].mean().item()


def compute_split_half_stability(
    pos_acts: torch.Tensor, neg_acts: torch.Tensor, n_splits: int = 20
) -> float:
    """Mean cosine similarity between steering vectors computed from random half-splits.

    pos_acts: [n_pairs, d_model]
    neg_acts: [n_pairs, d_model]
    n_splits: number of random splits to average over.
    Returns: mean cosine similarity in [-1, 1], higher means the vector is stable.
    """
    n = len(pos_acts)
    half = n // 2
    if half < 1:
        return 0.0

    sims: list[float] = []
    for _ in range(n_splits):
        perm = torch.randperm(n, device=pos_acts.device)
        a, b = perm[:half], perm[half : 2 * half]

        mean_a = (pos_acts[a] - neg_acts[a]).mean(dim=0)
        mean_b = (pos_acts[b] - neg_acts[b]).mean(dim=0)

        norm_a = mean_a.norm().clamp(min=1e-8)
        norm_b = mean_b.norm().clamp(min=1e-8)
        cos = torch.dot(mean_a / norm_a, mean_b / norm_b)
        sims.append(cos.item())

    return float(np.mean(sims))


def compute_snr(pos_activations: torch.Tensor, neg_activations: torch.Tensor) -> float:
    """Signal-to-noise ratio of per-pair difference vectors.

    pos_activations: [n_pairs, d_model]
    neg_activations: [n_pairs, d_model]
    Returns: ||mean(diffs)|| / std(||diffs||), higher means the shared signal dominates.
    """
    diffs = pos_activations - neg_activations
    mean_diff_norm = diffs.mean(dim=0).norm().item()
    per_pair_norms = diffs.norm(dim=1)
    std_norms = per_pair_norms.std().item()
    if std_norms < 1e-8:
        return 0.0
    return mean_diff_norm / std_norms


def compute_norm_sensitivity(pos_activations: torch.Tensor, neg_activations: torch.Tensor) -> float:
    """Cosine similarity between raw-mean and per-pair-normalized-mean steering directions.

    pos_activations: [n_pairs, d_model]
    neg_activations: [n_pairs, d_model]
    Returns: cosine similarity in [-1, 1]. Values near 1.0 mean magnitude variation
    doesn't distort the steering direction. Low values mean high-magnitude outlier
    pairs are pulling the raw mean away from the consensus direction.
    """
    diffs = pos_activations - neg_activations
    raw_mean = diffs.mean(dim=0)
    norms = diffs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized_mean = (diffs / norms).mean(dim=0)
    cos_sim = torch.nn.functional.cosine_similarity(
        raw_mean.unsqueeze(0), normalized_mean.unsqueeze(0),
    )
    return cos_sim.item()


def compute_steering_vector(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    behavior: str,
    layer: int,
    use_pca: bool = True,
    run_probe: bool = False,
) -> SteeringVector:
    """Compute steering vector from activation differences.

    pos_activations: [n, d_model]
    neg_activations: [n, d_model]
    use_pca: if True, use first principal component of per-pair diffs; else mean diff.
    run_probe: if True, train a logistic regression probe to measure separability.
    """
    diffs = pos_activations - neg_activations  # [n, d_model]

    consistency = compute_consistency(pos_activations, neg_activations) if diffs.shape[0] >= 2 else 0.0
    stability = compute_split_half_stability(pos_activations, neg_activations) if diffs.shape[0] >= 4 else 0.0
    snr = compute_snr(pos_activations, neg_activations) if diffs.shape[0] >= 2 else 0.0
    norm_sensitivity = compute_norm_sensitivity(pos_activations, neg_activations) if diffs.shape[0] >= 2 else 0.0

    # Compute PCA explained variance for top components (up to 10)
    n_components = 10
    explained_variances: tuple[float, ...] = ()
    if diffs.shape[0] >= 2:
        centered = diffs - diffs.mean(dim=0)
        q = min(n_components, diffs.shape[0])
        _, s, v = torch.pca_lowrank(centered, q=q)
        variances = s ** 2
        total_var = variances.sum()
        explained_variances = tuple((variances[i] / total_var).item() for i in range(len(variances)))

    if use_pca and diffs.shape[0] >= 2:
        direction = v[:, 0]  # first principal component, shape [d_model]

        # Ensure direction aligns with mean diff (not flipped)
        mean_diff = diffs.mean(dim=0)
        if torch.dot(direction, mean_diff) < 0:
            direction = -direction
    else:
        direction = diffs.mean(dim=0)

    # Capture pre-normalization magnitude relative to activation norms
    raw_magnitude = direction.norm().item()
    mean_act_norm = (
        (pos_activations.norm(dim=1).mean() + neg_activations.norm(dim=1).mean()) / 2
    ).item()
    normalized_magnitude = raw_magnitude / mean_act_norm if mean_act_norm > 0 else 0.0

    # Normalize to unit vector — alpha is scaled by activation norm at inference
    direction = direction / direction.norm()

    # Logistic regression probe for linear separability
    accuracy = 0.0
    if run_probe:
        accuracy = probe_layer(pos_activations, neg_activations)

    top3 = "  ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(explained_variances[:3]))
    log.info(
        "Layer %2d: magnitude=%.4f  %s  probe_acc=%.1f%%  consistency=%.3f  stability=%.3f  snr=%.2f  norm_sens=%.3f",
        layer, normalized_magnitude, top3, accuracy * 100, consistency, stability, snr, norm_sensitivity,
    )
    return SteeringVector(
        vector=direction, layer=layer, behavior=behavior,
        magnitude=normalized_magnitude, explained_variances=explained_variances,
        probe_accuracy=accuracy, consistency=consistency, stability=stability, snr=snr, norm_sensitivity=norm_sensitivity,
    )


def compute_steering_vectors_all_layers(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    behavior: str,
    use_pca: bool = True,
) -> list[SteeringVector]:
    """Compute steering vectors for every layer from multi-layer activations.

    pos_activations: [n_pairs, n_layers, d_model]
    neg_activations: [n_pairs, n_layers, d_model]
    Returns: list of SteeringVector, one per layer.
    """
    n_layers = pos_activations.shape[1]
    vectors: list[SteeringVector] = []

    for layer_idx in range(n_layers):
        pos_layer = pos_activations[:, layer_idx, :]  # [n_pairs, d_model]
        neg_layer = neg_activations[:, layer_idx, :]
        vec = compute_steering_vector(pos_layer, neg_layer, behavior, layer_idx, use_pca, run_probe=True)
        vectors.append(vec)

    return vectors


def _pc1(v: SteeringVector) -> float:
    """Return PC1 explained variance, or 0.0 if not available."""
    return v.explained_variances[0] if v.explained_variances else 0.0


def find_best_layer(
    vectors: list[SteeringVector],
) -> tuple[int, list[tuple[int, float, float, float]]]:
    """Find the best layer by PC1 explained variance and probe accuracy.

    Returns: (best_layer_index, all layers ranked by (pc1_variance, probe_accuracy) descending).
    """
    ranked = sorted(vectors, key=lambda v: (_pc1(v), v.probe_accuracy), reverse=True)
    best_vec = ranked[0]
    ranked_tuples = [(v.layer, v.magnitude, _pc1(v), v.probe_accuracy) for v in ranked]
    return best_vec.layer, ranked_tuples


def _resolve_layers(
    layers: str | list[int] | tuple[int, ...],
    all_vectors: list[SteeringVector],
    ranked: list[tuple[int, float, float, float]],
) -> list[int]:
    """Parse the layers arg into a list of layer indices.

    Accepts: "best", "all", "top-N", comma-separated ints like "10,15,20",
    or a list/tuple of ints (as Fire may parse them).
    Returns: sorted list of layer indices.
    """
    n_layers = len(all_vectors)

    # Fire may parse "10,15,20" as a tuple of ints
    if isinstance(layers, (list, tuple)):
        indices = [int(x) for x in layers]
        for idx in indices:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Layer {idx} out of range [0, {n_layers - 1}].")
        return sorted(indices)

    layers = str(layers).strip().lower()

    if layers == "best":
        return [ranked[0][0]]
    if layers == "all":
        return list(range(n_layers))
    if layers.startswith("top-"):
        n = int(layers[4:])
        return sorted(entry[0] for entry in ranked[:n])

    # Comma-separated integers
    try:
        indices = [int(x.strip()) for x in layers.split(",")]
    except ValueError:
        raise ValueError(f"Invalid layers arg: '{layers}'. Use 'best', 'all', 'top-N', or comma-separated ints.")
    for idx in indices:
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer {idx} out of range [0, {n_layers - 1}].")
    return sorted(indices)


# ── Stage 5: Inference with Steering ─────────────────────────────────────────


def _print_block(label: str, section: str, text: str) -> None:
    """Print a labeled, indented block for terminal readability."""
    header = f"┌─ {label} │ {section} "
    print(f"\n{header}{'─' * max(0, 80 - len(header))}")
    for line in text.splitlines():
        print(f"│  {line}")
    print(f"└{'─' * 79}")


def _set_seed(seed: int) -> None:
    """Set random seed for reproducible generation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_baseline(
    bundle: ModelBundle,
    prompt: str,
    max_new_tokens: int = 512,
    seed: int = 42,
) -> str:
    """Generate text without steering for comparison."""
    _set_seed(seed)
    formatted = _format_chat(bundle.tokenizer, prompt)
    _print_block("BASELINE", "Prompt", formatted)
    inputs = bundle.tokenizer(formatted, return_tensors="pt").to(bundle.device)
    with torch.no_grad():
        output_tokens = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
    # Decode only the new tokens (skip the prompt)
    new_tokens = output_tokens[0, inputs["input_ids"].shape[1] :]
    result = bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    _print_block("BASELINE", "Response", result)
    return result


def generate_with_steering(
    bundle: ModelBundle,
    prompt: str,
    vector: SteeringVector,
    alpha: float = 3.0,
    max_new_tokens: int = 512,
    seed: int = 42,
) -> str:
    """Generate text with a steering vector applied via hook.

    Alpha is relative to activation magnitude: alpha=0.5 means perturb by 50% of
    the activation norm. The steering vector is unit-normalized, so the actual
    perturbation magnitude is alpha * ||activation||.
    """
    _set_seed(seed)
    steering_vec = vector.vector.to(dtype=torch.bfloat16, device=bundle.device)

    def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> Any:  # noqa: A002
        """Add steering vector scaled by alpha * activation norm."""
        hidden = output[0] if isinstance(output, tuple) else output
        act_norm = hidden.norm(dim=-1, keepdim=True)
        hidden = hidden + alpha * act_norm * steering_vec
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    layer_module = _get_layer_module(bundle.model, vector.layer)
    handle = layer_module.register_forward_hook(hook_fn)

    try:
        formatted = _format_chat(bundle.tokenizer, prompt)
        _print_block(f"STEERED layer={vector.layer} α={alpha}", "Prompt", formatted)
        inputs = bundle.tokenizer(formatted, return_tensors="pt").to(bundle.device)
        with torch.no_grad():
            output_tokens = bundle.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
    finally:
        handle.remove()

    new_tokens = output_tokens[0, inputs["input_ids"].shape[1] :]
    result = bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    _print_block(f"STEERED layer={vector.layer} α={alpha}", "Response", result)
    return result


def generate_baseline_batch(
    bundle: ModelBundle,
    prompts: list[str],
    max_new_tokens: int = 512,
    seed: int = 42,
) -> list[str]:
    """Generate baseline text for multiple prompts in a single batched call."""
    _set_seed(seed)
    formatted = [_format_chat(bundle.tokenizer, p) for p in prompts]
    bundle.tokenizer.padding_side = "left"
    inputs = bundle.tokenizer(formatted, return_tensors="pt", padding=True).to(bundle.device)
    with torch.no_grad():
        output_tokens = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
    prompt_len = inputs["input_ids"].shape[1]
    results: list[str] = []
    for i in range(len(prompts)):
        new_tokens = output_tokens[i, prompt_len:]
        results.append(bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return results


def generate_with_steering_batch(
    bundle: ModelBundle,
    prompts: list[str],
    vector: SteeringVector,
    alpha: float = 3.0,
    max_new_tokens: int = 512,
    seed: int = 42,
) -> list[str]:
    """Generate steered text for multiple prompts in a single batched call."""
    _set_seed(seed)
    steering_vec = vector.vector.to(dtype=torch.bfloat16, device=bundle.device)

    def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> Any:  # noqa: A002
        hidden = output[0] if isinstance(output, tuple) else output
        act_norm = hidden.norm(dim=-1, keepdim=True)
        hidden = hidden + alpha * act_norm * steering_vec
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    layer_module = _get_layer_module(bundle.model, vector.layer)
    handle = layer_module.register_forward_hook(hook_fn)

    try:
        formatted = [_format_chat(bundle.tokenizer, p) for p in prompts]
        bundle.tokenizer.padding_side = "left"
        inputs = bundle.tokenizer(formatted, return_tensors="pt", padding=True).to(bundle.device)
        with torch.no_grad():
            output_tokens = bundle.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
    finally:
        handle.remove()

    prompt_len = inputs["input_ids"].shape[1]
    results: list[str] = []
    for i in range(len(prompts)):
        new_tokens = output_tokens[i, prompt_len:]
        results.append(bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return results


# ── Pipeline Entry Point ─────────────────────────────────────────────────────


def run(
    behavior: str,
    model_name: str = "Qwen/Qwen3.5-9B",
    alphas: str = "1.0",
    layers: str = "best",
    eval_prompts_override: str = "",
    n_eval_prompts: int = 1,
    use_pca: bool = True,
    seed: int = 42,
    remote: bool = False,
) -> None:
    """Run the full autosteer pipeline end-to-end."""
    bdir = _behavior_dir(behavior)
    pairs_path = bdir / "pairs.json"

    # Stage 1: Load contrastive pairs
    pairs = _load_pairs(pairs_path)
    log.info("Stage 1: Loaded %d contrastive pairs from %s", len(pairs), pairs_path)

    method = "pca" if use_pca else "mean"
    mslug = _model_slug(model_name)

    # Stage 2: Load model (or just config when remote — avoids loading 18GB of weights)
    bundle: ModelBundle | None = None
    if remote:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        text_config = getattr(config, "text_config", config)
        n_layers: int = text_config.num_hidden_layers
        log.info("Stage 2: Remote mode — loaded config only (%d layers)", n_layers)
    else:
        log.info("Stage 2: Loading model...")
        bundle = load_model(model_name)
        n_layers = bundle.n_layers

    # Always compute vectors for all layers
    cached_vectors: list[SteeringVector | None] = [
        _load_vector(bdir / "vectors" / f"{mslug}_layer{l}_{method}.pt")
        for l in range(n_layers)
    ]
    if all(v is not None for v in cached_vectors):
        all_vectors: list[SteeringVector] = cached_vectors  # type: ignore[assignment]
        log.info("Stage 3+4: Loaded %d cached steering vectors", len(all_vectors))
    else:
        if remote:
            raise RuntimeError(
                "Steering vectors not cached — run locally first to compute them:\n"
                f"  python run.py run --behavior={behavior}"
            )
        assert bundle is not None
        log.info("Stage 3: Extracting activations across ALL %d layers...", n_layers)
        pos_acts, neg_acts = extract_activations_all_layers(bundle, pairs)

        log.info("Stage 4: Computing steering vectors for all layers (%s)...", method)
        all_vectors = compute_steering_vectors_all_layers(pos_acts, neg_acts, behavior, use_pca)

    best_layer, ranked_layers = find_best_layer(all_vectors)
    log.info("=== Layer Ranking (top 10 by explained variance) ===")
    for rank, (l, mag, ev, acc) in enumerate(ranked_layers[:10]):
        log.info(
            "  #%d layer %2d: explained_var=%.1f%%  magnitude=%.4f  probe_acc=%.1f%%",
            rank + 1, l, ev * 100, mag, acc * 100,
        )

    # Save vectors for all layers (skip if loaded from cache)
    if not all(v is not None for v in cached_vectors):
        for v in all_vectors:
            vpath = bdir / "vectors" / f"{mslug}_layer{v.layer}_{method}.pt"
            _save_vector(v, vpath)
        log.info("Saved steering vectors for all %d layers", len(all_vectors))

    layer_stats = [
        {
            "layer": v.layer, "magnitude": v.magnitude,
            "explained_variances": list(v.explained_variances),
            "probe_accuracy": v.probe_accuracy,
            "consistency": v.consistency, "stability": v.stability, "snr": v.snr,
            "norm_sensitivity": v.norm_sensitivity,
        }
        for v in all_vectors
    ]

    # Resolve which layers to run inference on
    selected_layers = _resolve_layers(layers, all_vectors, ranked_layers)
    log.info("Selected %d layer(s) for inference: %s", len(selected_layers), selected_layers)

    # Parse alpha values — sweep if multiple provided
    if isinstance(alphas, (list, tuple)):
        alpha_values = [float(a) for a in alphas]
    else:
        alpha_values = [float(a) for a in str(alphas).split(",")]
    if eval_prompts_override:
        eval_prompts = [p.strip() for p in eval_prompts_override.split("|")]
    else:
        eval_prompts = EVAL_PROMPTS[:n_eval_prompts]

    # Stage 5: Generate baselines + steered outputs
    baselines_path = bdir / "baselines" / f"{mslug}_seed{seed}.json"
    baselines_path.parent.mkdir(parents=True, exist_ok=True)

    baselines_cache: dict[str, str] = {}
    if baselines_path.exists():
        baselines_cache = json.loads(baselines_path.read_text())

    all_steered: list[dict[str, object]] = []

    if remote:
        import time as _time

        from modal_app import app as modal_app, SteeringInference

        # Serialize vectors for selected layers
        vector_bytes_by_layer: dict[int, bytes] = {
            layer_idx: serialize_vector(all_vectors[layer_idx])
            for layer_idx in selected_layers
        }

        n_tasks = len(selected_layers) * len(alpha_values) * len(eval_prompts)
        log.info(
            "Connecting to Modal — %d steered generations (%d layers x %d alphas x %d prompts)...",
            n_tasks, len(selected_layers), len(alpha_values), len(eval_prompts),
        )

        t0 = _time.monotonic()
        with modal_app.run():
            inference = SteeringInference()
            result = inference.run_sweep.remote(
                eval_prompts, vector_bytes_by_layer, alpha_values, seed,
            )

        elapsed = _time.monotonic() - t0
        log.info("Modal sweep done in %.1fs (%.1fs/task)", elapsed, elapsed / max(n_tasks, 1))

        baseline_outputs = result["baselines"]
        all_steered = result["steered"]

        # Update baseline cache
        for prompt, output in zip(eval_prompts, baseline_outputs):
            baselines_cache[prompt] = output
        baselines_path.write_text(json.dumps(baselines_cache, indent=2))
    else:
        assert bundle is not None
        baseline_outputs: list[str] = []
        changed = False
        for i, prompt in enumerate(eval_prompts):
            if prompt in baselines_cache:
                baseline_outputs.append(baselines_cache[prompt])
            else:
                log.info("Generating baseline %d/%d: %s", i + 1, len(eval_prompts), prompt[:50])
                baseline = generate_baseline(bundle, prompt, seed=seed)
                baseline_outputs.append(baseline)
                baselines_cache[prompt] = baseline
                changed = True

        if changed:
            baselines_path.write_text(json.dumps(baselines_cache, indent=2))
        log.info("Stage 5: %d baseline outputs ready (%d cached)", len(baseline_outputs), len(baselines_cache))

        # Stage 5b: Generate steered outputs for each layer × alpha combination
        for layer_idx in selected_layers:
            vector = all_vectors[layer_idx]
            for a in alpha_values:
                log.info("── Generating steered outputs for layer %d, α=%.1f (seed=%d) ──", layer_idx, a, seed)
                steered_outputs: list[str] = []
                for i, prompt in enumerate(eval_prompts):
                    log.info("Generating steered %d/%d (layer=%d, α=%.1f)", i + 1, len(eval_prompts), layer_idx, a)
                    steered = generate_with_steering(bundle, prompt, vector, a, seed=seed)
                    steered_outputs.append(steered)

                all_steered.append({
                    "layer": layer_idx,
                    "alpha": a,
                    "outputs": steered_outputs,
                })

    # Save run log — append to existing results file or create new one
    run_path = bdir / "runs" / f"{mslug}_{method}.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)

    run_entry: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "layers": selected_layers,
        "n_pairs": len(pairs),
        "use_pca": use_pca,
        "seed": seed,
        "eval_prompts": eval_prompts,
        "baselines": baseline_outputs,
        "steered": all_steered,
    }

    existing_data: dict[str, object] = {"layer_stats": [], "runs": []}
    if run_path.exists():
        existing_data = json.loads(run_path.read_text())
        # Migrate old array format to new object format
        if isinstance(existing_data, list):
            existing_data = {"layer_stats": [], "runs": existing_data}

    existing_data["layer_stats"] = layer_stats
    existing_data["runs"].append(run_entry)  # type: ignore[union-attr]
    run_path.write_text(json.dumps(existing_data, indent=2))
    log.info("Run log saved to %s", run_path)


def main() -> None:
    """CLI entry point."""
    fire.Fire(run)


if __name__ == "__main__":
    main()
