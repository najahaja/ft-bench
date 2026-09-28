# FT-Bench: QLoRA Fine-Tuning vs AWQ Quantization Benchmark

A rigorous end-to-end benchmark comparing zero-shot inference, QLoRA fine-tuning,
and AWQ 4-bit quantization of Llama-3.2-3B-Instruct on a real-world NLU task.

## What This Project Does

Evaluates three systems on NLU (intent detection + slot filling) using the MASSIVE
dataset (en-US, 2,974 test samples, 60 intents, 55 slot types):

| System | Description |
|--------|-------------|
| System A - Base | meta-llama/Llama-3.2-3B-Instruct zero-shot (no fine-tuning) |
| System B - Fine-Tuned | System A + QLoRA on 11,481 MASSIVE training samples |
| System C - AWQ | System B compressed to INT4 using AWQ quantization |

---

## Results

### Accuracy (2,974 test samples)

| Metric | Base (A) | Fine-Tuned (B) | AWQ 4-bit (C) | A to B | B to C |
|--------|:--------:|:--------------:|:-------------:|:------:|:------:|
| Intent Accuracy | 41.26% | 88.77% | 88.53% | +47.5pp | -0.24pp |
| Slot F1 | 16.41% | 85.34% | 85.99% | +68.9pp | +0.65pp |
| Exact Match | 2.62% | 71.69% | 72.19% | +69.1pp | +0.50pp |
| JSON Valid Rate | 99.53% | 99.53% | 99.50% | 0pp | -0.03pp |

### Key Findings

- Fine-tuning delivers 27x improvement in exact match (2.62% to 71.69%)
- AWQ 4-bit quantization preserves full accuracy with zero meaningful degradation
- AWQ slightly outperforms FP16 on Slot F1 (+0.65pp) and Exact Match (+0.50pp)
- Both fine-tuned models produce valid JSON over 99.5% of the time

### VRAM Usage

| System | Precision | VRAM |
|--------|-----------|------|
| System A / B | FP16 | 6.20 GB |
| System C | INT4 (AWQ) | 6.23 GB |

Note: VRAM is similar on T4 because FP16 activations are still needed.
On A100/H100, AWQ typically saves 40-60% VRAM.

### Error Analysis (4-Quadrant)

Base to Fine-Tuned (n=2,974):
- Both correct: 68 (2.3%)
- FT gains (Base wrong, FT right): 2,064 (69.4%)
- FT regressions: 10 (0.3%)
- Both wrong: 832 (28.0%)

Fine-Tuned to AWQ (n=2,974):
- Both correct: 2,080 (69.9%)
- AWQ gains: 67 (2.3%)
- AWQ regressions: 52 (1.7%)
- Both wrong: 775 (26.1%)

---

## Repository Structure

    ft-bench/
    |-- ftbench/            # Core Python package
    |-- data/               # MASSIVE dataset (auto-generated)
    |   |-- train.jsonl     # 11,481 samples
    |   |-- val.jsonl       # 2,033 samples
    |   +-- test.jsonl      # 2,974 samples
    |-- eval/results/
    |   |-- base/           # System A metrics + records
    |   |-- finetuned/      # System B metrics + records
    |   |-- quantized/      # System C metrics + records
    |   +-- error_analysis.json
    |-- docs/               # Design, decisions, failures, interview prep
    |-- scripts/            # Training and quantization scripts
    |-- tests/              # 34 passing unit tests
    +-- notebooks/          # Kaggle evaluation notebooks

---

## Methodology

### Dataset: MASSIVE (en-US)
- Source: Amazon Science MASSIVE dataset
- Task: Given utterance, output JSON with intent + slots
- Split: 11,481 train / 2,033 val / 2,974 test (after deduplication, leakage check)

### System A: Zero-Shot Baseline
- Model: meta-llama/Llama-3.2-3B-Instruct (FP16)
- No training, instruction-prompted only
- Batched inference: batch_size=8, max_new_tokens=128, greedy decoding

### System B: QLoRA Fine-Tuning
- QLoRA: 4-bit NF4 base + FP16 LoRA adapters
- LoRA config: rank=16, alpha=32, dropout=0.05, all linear layers
- Training: 3 epochs, lr=2e-4, effective batch=16
- Hardware: Kaggle T4 GPU (15GB VRAM)
- HuggingFace: https://huggingface.co/najahaja/ftbench-qlora-llama3.2-3b

### System C: AWQ 4-bit Quantization
- Method: Activation-aware Weight Quantization (INT4, group_size=128)
- Tool: llmcompressor + compressed-tensors
- HuggingFace: https://huggingface.co/najahaja/ftbench-qlora-llama3.2-3b-awq

---

## Quick Start

    git clone https://github.com/najahaja/ft-bench.git
    cd ft-bench
    pip install -e ".[dev]"
    pytest tests/ -v

---

## HuggingFace Models

| Model | Link |
|-------|------|
| Fine-Tuned FP16 | https://huggingface.co/najahaja/ftbench-qlora-llama3.2-3b |
| LoRA Adapter | https://huggingface.co/najahaja/ftbench-qlora-llama3.2-3b-adapter |
| AWQ 4-bit | https://huggingface.co/najahaja/ftbench-qlora-llama3.2-3b-awq |

---

## Author

Najah - https://github.com/najahaja
