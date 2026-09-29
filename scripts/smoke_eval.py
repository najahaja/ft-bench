"""
Smoke evaluation for CI — runs 5 samples without a GPU using mock generation.
Usage: python scripts/smoke_eval.py --n-samples 5 --system base
"""
import argparse
import json
import os
import random
from ftbench.common.io import read_jsonl, write_json
from ftbench.eval.metrics import compute_metrics


def mock_generate(prompt: str) -> str:
    """Return a plausible but random JSON output for smoke testing."""
    templates = [
        '{"intent": "atis_flight", "slots": {"fromloc.city_name": "boston", "toloc.city_name": "denver"}}',
        '{"intent": "atis_airfare", "slots": {"fromloc.city_name": "new york"}}',
        '{"intent": "atis_ground_service", "slots": {}}',
    ]
    return random.choice(templates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--system", default="base")
    args = parser.parse_args()

    out_dir = "eval/results/smoke"
    os.makedirs(out_dir, exist_ok=True)

    samples = read_jsonl("data/test.jsonl")[: args.n_samples]
    records = []
    for s in samples:
        from ftbench.prompts.templates import build_inference_prompt
        from ftbench.eval.parse import parse_output
        prompt = build_inference_prompt(s["utterance"])
        raw = mock_generate(prompt)
        parsed = parse_output(raw)
        records.append({
            "id": s.get("id", ""),
            "utterance": s["utterance"],
            "ground_truth": s["output"],
            "raw_output": raw,
            "prediction": parsed,
            "system": args.system,
        })

    metrics = compute_metrics(records)
    write_json({"system": args.system, "n_samples": len(records), "metrics": metrics},
               f"{out_dir}/metrics.json")
    print(f"[smoke] n={len(records)} metrics={metrics}")


if __name__ == "__main__":
    main()
