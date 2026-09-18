#!/usr/bin/env python3
"""Compute 95% bootstrap confidence intervals for all quality metrics from records.jsonl."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftbench.common.io import read_jsonl, read_json, write_json
from ftbench.eval.metrics import compute_metrics
from ftbench.eval.stats import bootstrap_ci


def main():
    parser = argparse.ArgumentParser(description="Compute bootstrap CIs for quality metrics")
    parser.add_argument("--records", default="eval/results/base/records.jsonl",
                        help="Path to records.jsonl")
    parser.add_argument("--metrics-file", default="eval/results/base/metrics.json",
                        help="Path to metrics.json to update")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.records):
        print(f"[!] Records file not found: {args.records}")
        sys.exit(1)

    records = read_jsonl(args.records)
    print(f"[+] Loaded {len(records)} records from {args.records}")

    print("[+] Computing bootstrap CIs (n=1000, 95% CI)...")
    intent_ci = bootstrap_ci(records, "intent_accuracy", n_bootstrap=args.n_bootstrap, ci=args.ci, seed=args.seed)
    slot_ci = bootstrap_ci(records, "slot_f1", n_bootstrap=args.n_bootstrap, ci=args.ci, seed=args.seed)
    em_ci = bootstrap_ci(records, "exact_match", n_bootstrap=args.n_bootstrap, ci=args.ci, seed=args.seed)

    print(f"  intent_accuracy : {intent_ci[0]}% (95% CI: [{intent_ci[1]}%, {intent_ci[2]}%])")
    print(f"  slot_f1         : {slot_ci[0]}% (95% CI: [{slot_ci[1]}%, {slot_ci[2]}%])")
    print(f"  exact_match     : {em_ci[0]}% (95% CI: [{em_ci[1]}%, {em_ci[2]}%])")

    if os.path.exists(args.metrics_file):
        data = read_json(args.metrics_file)
    else:
        data = {"n_samples": len(records), "metrics": compute_metrics(records)}

    data["intent_accuracy_95ci"] = {"point": intent_ci[0], "lower": intent_ci[1], "upper": intent_ci[2]}
    data["slot_f1_95ci"] = {"point": slot_ci[0], "lower": slot_ci[1], "upper": slot_ci[2]}
    data["exact_match_95ci"] = {"point": em_ci[0], "lower": em_ci[1], "upper": em_ci[2]}

    write_json(data, args.metrics_file)
    print(f"[✓] Updated {args.metrics_file} with bootstrap CIs.")


if __name__ == "__main__":
    main()