#!/usr/bin/env python3
"""AWQ 4-bit post-training quantization via llm-compressor (Kaggle T4 / GPU required).

Uses llm-compressor, NOT autoawq (archived May 2025).
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


def main():
    parser = argparse.ArgumentParser(description="FT-Bench: AWQ Quantization (llm-compressor)")
    parser.add_argument("--config", default="configs/quantization.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["quantization"]

    print("=" * 65)
    print("  FT-Bench: AWQ 4-Bit Quantization (llm-compressor)")
    print(f"  Source : {cfg['merged_model_path']}")
    print(f"  Output : {cfg['output_path']}")
    print(f"  Bits   : {cfg['w_bit']}-bit, group_size={cfg['q_group_size']}")
    print("=" * 65)

    if args.dry_run:
        print("[OK] Dry run — configuration validated.")
        return

    from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoTokenizer
    from ftbench.common.io import read_jsonl

    tokenizer = AutoTokenizer.from_pretrained(cfg["merged_model_path"])
    model = SparseAutoModelForCausalLM.from_pretrained(
        cfg["merged_model_path"],
        device_map="auto",
        torch_dtype="auto",
    )

    # Calibration data from training split
    calib_texts = []
    if os.path.exists(cfg.get("calibration_dataset", "")):
        records = read_jsonl(cfg["calibration_dataset"])
        calib_texts = [r["prompt"] for r in records[: cfg.get("n_calibration_samples", 512)]]

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    )

    oneshot(
        model=model,
        dataset=calib_texts,
        recipe=recipe,
        max_seq_length=cfg.get("max_seq_length", 512),
        num_calibration_samples=cfg.get("n_calibration_samples", 512),
    )

    model.save_pretrained(cfg["output_path"])
    tokenizer.save_pretrained(cfg["output_path"])
    print(f"[OK] Quantized model saved to {cfg['output_path']}")


if __name__ == "__main__":
    main()
