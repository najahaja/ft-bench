#!/usr/bin/env python3
"""FT-Bench: Baseline evaluation (System A — zero-shot, no fine-tuning).

Requires GPU. On Kaggle T4, uses the real Llama 3.2 3B Instruct model.
Use --mock for a CPU smoke test (validates pipeline end-to-end without a GPU).
"""
import argparse
import json
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from ftbench.common.io import read_jsonl, write_json
from ftbench.common.seed import seed_everything
from ftbench.eval.runner import run_eval
from ftbench.eval.metrics import compute_metrics
from ftbench.eval.stats import bootstrap_ci


def main():
    parser = argparse.ArgumentParser(description="FT-Bench: Baseline Evaluation (System A)")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="eval/results/base")
    parser.add_argument("--mock", action="store_true",
                        help="CPU smoke test — random NLUParse outputs, no GPU needed")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["evaluation"]

    seed_everything(cfg["seed"])
    test_file = cfg.get("test_file", "data/test.jsonl")

    if not os.path.exists(test_file):
        print(f"[!] {test_file} not found. Run scripts/prepare_dataset.py first.")
        sys.exit(1)

    samples = read_jsonl(test_file)
    if args.num_samples:
        samples = samples[:args.num_samples]
    print(f"[+] Evaluating {len(samples)} test samples (system=base)")

    if args.mock:
        # Load vocab for realistic mock outputs
        with open("configs/label_vocab.json") as f:
            vocab = json.load(f)
        intents = vocab.get("intents", ["unknown"])
        slot_types = vocab.get("slots", [])

        def generate_fn(prompt: str) -> str:
            if random.random() < 0.22:
                return "Here is my analysis of the utterance..."  # simulate parse failure
            intent = random.choice(intents)
            # Randomly pick 0–2 slot types for mock output
            n_slots = random.randint(0, min(2, len(slot_types)))
            slots = {st: "example" for st in random.sample(slot_types, n_slots)} if slot_types else {}
            return json.dumps({"intent": intent, "slots": slots})
    else:
        from transformers import AutoTokenizer, pipeline
        import torch
        with open("configs/serving.yaml") as f:
            s_cfg = yaml.safe_load(f)
        model_id = s_cfg["serving"]["models"]["base"]["id"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline(
            "text-generation", model=model_id, tokenizer=tokenizer,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg["temperature"],
            do_sample=cfg["temperature"] > 0,
            return_full_text=False,
        )
        def generate_fn(prompt: str) -> str:
            out = pipe(prompt)
            return out[0]["generated_text"] if out else ""

    os.makedirs(args.output_dir, exist_ok=True)
    records = run_eval(
        samples, generate_fn, system_name="base",
        output_path=os.path.join(args.output_dir, "records.jsonl"),
        verbose=True,
    )
    metrics = compute_metrics(records)
    intent_ci = bootstrap_ci(
        records, "intent_accuracy",
        n_bootstrap=cfg["bootstrap_n"], ci=cfg["ci"],
    )
    result = {
        "system": "base",
        "n_samples": len(records),
        "metrics": metrics,
        "intent_accuracy_95ci": {
            "point": intent_ci[0], "lower": intent_ci[1], "upper": intent_ci[2],
        },
    }
    write_json(result, os.path.join(args.output_dir, "metrics.json"))
    print("\n" + "=" * 60 + "\n  BASELINE RESULTS (System A)\n" + "=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
