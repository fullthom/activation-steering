"""Evaluate steered outputs using Claude Haiku as a judge."""

import asyncio
import json
import logging
from pathlib import Path

import anthropic
import dotenv
import pydantic

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"


class Grade(pydantic.BaseModel):
    """Scores for a single AI response."""

    coherence: int = pydantic.Field(description="1 = nonsense/garbled, 5 = fully coherent and well-structured")
    sutro: int = pydantic.Field(description="1 = no mention, 5 = mentions Sutro Tower")


async def grade_response(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    response: str,
) -> Grade:
    """Call Haiku to grade a single (prompt, response) pair."""
    message = await client.messages.create(
        model=MODEL,
        max_tokens=128,
        tools=[{"name": "grade_response", "description": "Grade an AI response.", "input_schema": Grade.model_json_schema()}],
        tool_choice={"type": "tool", "name": "grade_response"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"You are grading AI responses along two dimensions: coherence and mentioning of Sutro Tower in San Francisco. Read the user question, AI response, and return a score.\n\n"
                    f"User question: {prompt}\n\n"
                    f"AI response: {response}"
                ),
            }
        ],
    )
    for block in message.content:
        if block.type == "tool_use":
            return Grade.model_validate(block.input)
    raise ValueError("No tool use in response")


async def evaluate(run_file: str) -> None:
    """Evaluate the last run in a run file and write scored results."""
    path = Path(run_file)
    data = json.loads(path.read_text())
    run = data["runs"][-1]

    eval_prompts: list[str] = run["eval_prompts"]
    baselines: list[str] = run["baselines"]
    steered: list[dict] = run["steered"]

    client = anthropic.AsyncAnthropic()

    # Grade baselines
    log.info("Grading %d baselines...", len(baselines))
    baseline_scores = await asyncio.gather(*[
        grade_response(client, p, r) for p, r in zip(eval_prompts, baselines)
    ])

    baseline_results = [
        {"prompt": p, "response": r, **grade.model_dump()}
        for p, r, grade in zip(eval_prompts, baselines, baseline_scores)
    ]

    # Grade steered outputs
    total = sum(len(entry["outputs"]) for entry in steered)
    log.info("Grading %d steered outputs (%d combos)...", total, len(steered))

    # Build flat task list: (steered_idx, prompt_idx, coroutine)
    task_keys: list[tuple[int, int]] = []
    coros = []
    for si, entry in enumerate(steered):
        for pi, output in enumerate(entry["outputs"]):
            task_keys.append((si, pi))
            coros.append(grade_response(client, eval_prompts[pi], output))

    scores = await asyncio.gather(*coros)

    # Reassemble into per-(layer, alpha) groups
    score_map: dict[int, list[dict]] = {}
    for (si, pi), grade in zip(task_keys, scores):
        if si not in score_map:
            score_map[si] = [{}] * len(eval_prompts)
        score_map[si][pi] = {
            "prompt": eval_prompts[pi],
            "response": steered[si]["outputs"][pi],
            **grade.model_dump(),
        }

    steered_results = [
        {"layer": entry["layer"], "alpha": entry["alpha"], "results": score_map[si]}
        for si, entry in enumerate(steered)
    ]

    output = {
        "model": run["model_name"],
        "eval_prompts": eval_prompts,
        "baselines": baseline_results,
        "steered": steered_results,
    }

    out_path = path.with_name(path.stem + "_scored.json")
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <run_file.json>")
        sys.exit(1)
    asyncio.run(evaluate(sys.argv[1]))
