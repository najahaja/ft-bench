#!/usr/bin/env python3
"""CLI entry point for dataset preparation. Thin wrapper around ftbench.data.prepare.

Uses AmazonScience/massive (en-US) with official splits.
Run with --synthetic for a quick CPU smoke test (no HF download needed).
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftbench.data.prepare import prepare


def main():
    parser = argparse.ArgumentParser(description="FT-Bench: Dataset Preparation (MASSIVE)")
    parser.add_argument("--output-dir", default="data", help="Where to write train/val/test JSONL")
    parser.add_argument("--config", default="en-US", help="MASSIVE language config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (no HF download). CPU smoke test only. NOT for real results.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Truncate train split (smoke-test mode only)",
    )
    args = parser.parse_args()

    if args.synthetic:
        print("WARNING: SYNTHETIC DATA — NOT FOR REAL RESULTS")

    paths = prepare(
        output_dir=args.output_dir,
        config=args.config,
        seed=args.seed,
        use_synthetic=args.synthetic,
        max_samples=args.max_samples,
    )
    print("[OK] Dataset ready:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
