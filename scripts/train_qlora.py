#!/usr/bin/env python3
"""
Phase 5: QLoRA Fine-Tuning (System B) on MASSIVE dataset.

Follows the exact hyperparameter specification in configs/training.yaml
and docs/01-experiment-design.md.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Workaround for upstream TRL bug where _patch_chunked_ce_lm_head crashes on functools.partial
try:
    import trl.trainer.sft_trainer
    trl.trainer.sft_trainer._patch_chunked_ce_lm_head = lambda *args, **kwargs: None
except Exception:
    pass

try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
    except ImportError:
        class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
            """Self-contained completion-only loss collator for trl >= 0.20."""
            def __init__(self, response_template, tokenizer, mlm=False, ignore_index=-100):
                super().__init__(tokenizer=tokenizer, mlm=mlm)
                if isinstance(response_template, str):
                    self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
                else:
                    self.response_token_ids = response_template
                self.ignore_index = ignore_index

            def torch_call(self, examples):
                batch = super().torch_call(examples)
                for i in range(len(examples)):
                    labels = batch["labels"][i]
                    r_len = len(self.response_token_ids)
                    for j in range(len(labels) - r_len + 1):
                        if labels[j : j + r_len].tolist() == self.response_token_ids:
                            labels[: j + r_len] = self.ignore_index
                            break
                return batch

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None

from ftbench.common.gpu import get_gpu_stats, peak_vram_mb
from ftbench.common.io import read_jsonl, write_json
from ftbench.common.seed import seed_everything
from ftbench.prompts.templates import build_training_prompt
from scripts.merge_adapter import merge_and_save


def load_env_vars():
    """Load environment variables from .env if present."""
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class WandbGPUMonitorCallback(TrainerCallback):
    """Logs GPU memory, epoch wall-clock duration, and metrics to W&B / console."""
    def __init__(self, use_wandb: bool = False):
        self.use_wandb = use_wandb
        self.epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        duration_s = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
        gpu_stats = get_gpu_stats() or {}
        vram_peak = peak_vram_mb()

        log_payload = {
            "epoch": state.epoch,
            "epoch_duration_s": round(duration_s, 2),
            "vram_peak_mb": round(vram_peak, 2) if vram_peak else None,
            **gpu_stats,
        }

        print(f"[+] Epoch {state.epoch:.1f} completed in {duration_s:.1f}s | Peak VRAM: {vram_peak} MB")

        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(log_payload, step=state.global_step)
            except Exception as e:
                print(f"[!] Failed to log GPU stats to wandb: {e}")


def prepare_hf_dataset(
    jsonl_path: str,
    tokenizer,
    label_vocab: dict,
) -> Dataset:
    """Format raw dataset using ftbench.prompts.templates.build_training_prompt."""
    samples = read_jsonl(jsonl_path)
    formatted_texts = []
    for s in samples:
        text = build_training_prompt(
            utterance=s["utterance"],
            ground_truth=s["ground_truth"],
            tokenizer=tokenizer,
            label_vocab=label_vocab,
        )
        formatted_texts.append(text)

    return Dataset.from_dict({"text": formatted_texts})


def main():
    parser = argparse.ArgumentParser(description="FT-Bench Phase 5: QLoRA Fine-Tuning")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--hub-repo-id", default=None, help="Override Hub repo ID")
    parser.add_argument("--hf-token", default=None, help="Hugging Face access token")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if exists")
    args = parser.parse_args()

    load_env_vars()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Runtime copy of config to persist in train_metadata.json
    runtime_config = copy.deepcopy(cfg)

    m_cfg = cfg["model"]
    l_cfg = cfg["lora"]
    t_cfg = cfg["training"]
    c_cfg = cfg.get("checkpointing", {})
    d_cfg = cfg.get("data", {})

    seed_everything(t_cfg.get("seed", 42))

    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not hf_token:
        try:
            from huggingface_hub import get_token
            hf_token = get_token()
        except Exception:
            pass

    if not hf_token:
        print("[!] ERROR: No Hugging Face token detected! Pass --hf-token or set HF_TOKEN environment variable.")
    else:
        print(f"[+] Hugging Face token detected: {hf_token[:4]}...{hf_token[-4:]}")
    hub_repo_id = args.hub_repo_id or os.environ.get("HF_REPO_ID") or c_cfg.get("hub_repo_id")
    if hub_repo_id and "${" in hub_repo_id:
        hub_repo_id = os.environ.get("HF_REPO_ID", None)

    wandb_key = os.environ.get("WANDB_API_KEY")
    use_wandb = bool(wandb_key) and t_cfg.get("report_to") == "wandb"
    if use_wandb:
        try:
            import wandb
            wandb.login(key=wandb_key)
            wandb.init(
                project="ftbench",
                name=t_cfg.get("run_name", "qlora-llama3.2-3b"),
                config=cfg,
            )
        except Exception as e:
            print(f"[!] W&B init warning: {e}. Falling back to none.")
            use_wandb = False

    # 1. Tokenizer
    model_id = m_cfg["base_model"]
    print(f"[+] Loading tokenizer for: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Dataset preparation
    train_path = d_cfg.get("train_path", "data/train.jsonl")
    val_path = d_cfg.get("val_path", "data/val.jsonl")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"[!] Data files not found at {train_path}. Running prepare_dataset automatically...")
        from ftbench.data.prepare import prepare
        prepare(output_dir="data", config="en-US", seed=t_cfg.get("seed", 42))
        print("[✓] MASSIVE dataset prepared successfully.")

    vocab_path = "configs/label_vocab.json"
    with open(vocab_path) as f:
        label_vocab = json.load(f)

    print(f"[+] Formatting datasets using canonical build_training_prompt()...")
    train_dataset = prepare_hf_dataset(train_path, tokenizer, label_vocab)
    val_dataset = prepare_hf_dataset(d_cfg.get("val_path", "data/val.jsonl"), tokenizer, label_vocab)
    print(f"[+] Train dataset: {len(train_dataset)} examples | Val dataset: {len(val_dataset)} examples")

    # 3. Model with 4-bit quantization
    compute_dtype = getattr(torch, m_cfg.get("bnb_4bit_compute_dtype", "float16"))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=m_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=m_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=m_cfg.get("bnb_4bit_use_double_quant", True),
    )

    print(f"[+] Loading base model {model_id} in 4-bit (NF4, compute={compute_dtype})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        token=hf_token,
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # 4. LoRA Configuration
    lora_config = LoraConfig(
        r=l_cfg["r"],
        lora_alpha=l_cfg["lora_alpha"],
        lora_dropout=l_cfg["lora_dropout"],
        target_modules=l_cfg["target_modules"],
        bias=l_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 5. Data Collator for Completion-Only Loss Masking
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # 6. Training Arguments
    output_dir = c_cfg.get("output_dir", "/kaggle/working/checkpoints")
    push_to_hub = bool(hub_repo_id and c_cfg.get("push_to_hub_every_epoch", False))

    training_kwargs = dict(
        output_dir=output_dir,
        seed=t_cfg.get("seed", 42),
        num_train_epochs=t_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=t_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(t_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(t_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(t_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(t_cfg.get("max_grad_norm", 0.3)),
        fp16=t_cfg.get("fp16", True),
        bf16=False,
        logging_steps=t_cfg.get("logging_steps", 10),
        eval_strategy=t_cfg.get("eval_strategy", "epoch"),
        save_strategy=t_cfg.get("save_strategy", "epoch"),
        save_total_limit=3,
        load_best_model_at_end=t_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=t_cfg.get("metric_for_best_model", "eval_loss"),
        report_to="wandb" if use_wandb else "none",
        run_name=t_cfg.get("run_name", "qlora-llama3.2-3b"),
        push_to_hub=push_to_hub,
        hub_model_id=hub_repo_id if push_to_hub else None,
        hub_strategy="every_save" if push_to_hub else "end",
        hub_token=hf_token if push_to_hub else None,
    )

    import inspect
    max_len = t_cfg.get("max_seq_len", 640)

    if SFTConfig is not None:
        sft_init_params = inspect.signature(SFTConfig.__init__).parameters
        if "max_length" in sft_init_params:
            training_kwargs["max_length"] = max_len
        elif "max_seq_length" in sft_init_params:
            training_kwargs["max_seq_length"] = max_len

        if "dataset_text_field" in sft_init_params:
            training_kwargs["dataset_text_field"] = "text"

        sft_config = SFTConfig(**training_kwargs)

        trainer_params = inspect.signature(SFTTrainer.__init__).parameters
        trainer_kwargs = {
            "model": model,
            "args": sft_config,
            "train_dataset": train_dataset,
            "eval_dataset": val_dataset,
            "data_collator": collator,
            "callbacks": [WandbGPUMonitorCallback(use_wandb=use_wandb)],
        }
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = SFTTrainer(**trainer_kwargs)
    else:
        train_args = TrainingArguments(**training_kwargs)
        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=max_len,
            data_collator=collator,
            tokenizer=tokenizer,
            callbacks=[WandbGPUMonitorCallback(use_wandb=use_wandb)],
        )

    # 7. Check for checkpoint resumption
    resume_checkpoint = None
    if args.resume or os.path.isdir(output_dir):
        checkpoints = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ] if os.path.exists(output_dir) else []
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            resume_checkpoint = checkpoints[-1]
            print(f"[+] Found existing checkpoint: {resume_checkpoint} — resuming training.")

    # 8. Train
    t0_train = time.time()
    print("[+] Starting QLoRA fine-tuning...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    total_train_time_s = time.time() - t0_train
    print(f"[✓] Training completed in {total_train_time_s / 60:.2f} minutes.")

    # 9. Evaluate final loss
    eval_result = trainer.evaluate()
    print(f"[+] Final Validation Metrics: {eval_result}")

    # 10. Save best adapter
    best_adapter_dir = os.path.join(output_dir, "final_adapter")
    print(f"[+] Saving best adapter to: {best_adapter_dir}")
    trainer.model.save_pretrained(best_adapter_dir)
    tokenizer.save_pretrained(best_adapter_dir)

    if hub_repo_id:
        try:
            print(f"[+] Pushing final adapter to HF Hub: {hub_repo_id}...")
            trainer.model.push_to_hub(hub_repo_id, token=hf_token)
            tokenizer.push_to_hub(hub_repo_id, token=hf_token)
            print(f"[✓] Final adapter pushed to HF Hub.")
        except Exception as e:
            print(f"[!] Warning: failed to push adapter to hub: {e}")

    # 11. Produce merged FP16 model
    merged_output_dir = "models/finetuned"
    print(f"[+] Merging adapter into base model (FP16)...")
    try:
        merge_and_save(
            base_model_id=model_id,
            adapter_path=best_adapter_dir,
            output_dir=merged_output_dir,
            hub_repo_id=f"{hub_repo_id}-merged" if hub_repo_id else None,
            hf_token=hf_token,
        )
    except Exception as e:
        print(f"[!] Warning: Merge step failed or deferred: {e}")

    # 12. Save metadata artifact
    meta_dir = "eval/results/finetuned"
    os.makedirs(meta_dir, exist_ok=True)
    metadata = {
        "system": "finetuned",
        "git_commit": get_git_commit_hash(),
        "total_training_time_s": round(total_train_time_s, 2),
        "train_loss": train_result.training_loss if hasattr(train_result, "training_loss") else None,
        "eval_loss": eval_result.get("eval_loss"),
        "metrics": eval_result,
        "config": runtime_config,
    }
    meta_path = os.path.join(meta_dir, "train_metadata.json")
    write_json(metadata, meta_path)
    print(f"[✓] Training metadata saved to {meta_path}")


if __name__ == "__main__":
    main()