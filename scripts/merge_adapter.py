#!/usr/bin/env python3
"""Merge LoRA adapter weights into base model (FP16) for evaluation and deployment."""
import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def merge_and_save(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
    hub_repo_id: str = None,
    hf_token: str = None,
):
    print(f"[+] Loading base model: {base_model_id} in float16...")
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        token=hf_token,
    )

    print(f"[+] Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("[+] Merging adapter weights into base model...")
    merged_model = model.merge_and_unload()

    print(f"[+] Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"[✓] Merged model successfully saved to {output_dir}")

    if hub_repo_id:
        print(f"[+] Pushing merged model to HF Hub: {hub_repo_id}...")
        merged_model.push_to_hub(hub_repo_id, token=hf_token, safe_serialization=True)
        tokenizer.push_to_hub(hub_repo_id, token=hf_token)
        print(f"[✓] Pushed to HF Hub: {hub_repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--adapter-path", required=True, help="Path to trained LoRA adapter directory")
    parser.add_argument("--output-dir", default="models/finetuned", help="Target directory for merged model")
    parser.add_argument("--hub-repo-id", default=None, help="Optional HF repo ID to push merged model")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token")
    args = parser.parse_args()

    merge_and_save(
        base_model_id=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        hub_repo_id=args.hub_repo_id,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()