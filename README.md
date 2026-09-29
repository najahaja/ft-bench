# FT-Bench: Llama 3.2 3B Benchmark for Fine-Tuning vs Quantization

> End-to-end benchmark comparing zero-shot inference, QLoRA fine-tuning, and AWQ 4-bit quantization for Llama-3.2-3B-Instruct on a real-world NLU task.

[![CI](https://github.com/najahaja/ft-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/najahaja/ft-bench/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/model-Llama--3.2--3B--Instruct-orange)](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
[![Copyright](https://img.shields.io/badge/copyright-2026-najahaja-lightgrey)](https://github.com/najahaja)

---

## Overview

FT-Bench is a practical benchmarking project for evaluating how small language models perform under three settings:

- Zero-shot inference
- QLoRA fine-tuning
- AWQ 4-bit quantization

The project focuses on intent detection and slot-filling for a real-world dataset: MASSIVE (n = 2,974 test samples), using Llama-3.2-3B-Instruct as the base model.

This repository is designed to help researchers and engineers compare model quality, deployment trade-offs, and efficiency in a clean and reproducible setup.

---

## Why This Project Matters

This project demonstrates that:

- Fine-tuning can dramatically improve task performance
- Quantization can preserve accuracy while reducing inference cost
- Real-world NLU benchmarks are useful for validating practical deployment choices
- A compact 3B model can become highly effective with proper adaptation

---

## Key Results

Benchmark results on the test set (n = 2,974 samples):

| Metric | Base (Zero-Shot) | Fine-Tuned (QLoRA) | AWQ (4-bit) | Δ Base → FT | Δ FT → AWQ |
|---|---:|---:|---:|---:|---:|
| Intent Accuracy | 41.26% | 88.77% | 88.53% | +47.5 pp | -0.24 pp |
| Slot F1 | 16.41% | 85.34% | 85.99% | +68.9 pp | +0.65 pp |
| Exact Match | 2.62% | 71.69% | 72.19% | +69.1 pp | +0.50 pp |
| JSON Valid Rate | 99.53% | 99.53% | 99.50% | — | -0.03 pp |

### Important takeaways

- Fine-tuning delivers a huge improvement in exact match: 2.62% → 71.69%
- AWQ quantization preserves nearly all accuracy while reducing model size and runtime cost
- Both fine-tuned and quantized versions keep JSON validity above 99.5%
- The benchmark shows a strong case for practical model adaptation rather than using a base model alone

---

## Architecture

```text
System A — Base          meta-llama/Llama-3.2-3B-Instruct  (zero-shot, fp16)
System B — Fine-Tuned    najahaja/ftbench-qlora-llama3.2-3b (QLoRA, fp16)
System C — AWQ           najahaja/ftbench-qlora-llama3.2-3b-awq (int4, AWQ)
```

### Training setup

- Dataset: MASSIVE en-US
- Task: intent classification + slot filling
- Train samples: 11,481
- Validation samples: 2,033
- Test samples: 2,974
- Method: QLoRA (r=16, α=32) across all linear layers
- Hardware: NVIDIA T4 16GB (Kaggle)
- Training time: ~45 minutes

---

## Project Structure

```text
ft-bench/
├── ftbench/                  # Core Python package
│   ├── common/               # Seed and I/O utilities
│   ├── eval/                 # Metrics, parsing, statistics, runner
│   └── prompts/              # Prompt templates
├── data/                     # MASSIVE dataset (auto-generated)
│   ├── train.jsonl           # 11,481 samples
│   ├── val.jsonl             # 2,033 samples
│   └── test.jsonl            # 2,974 samples
├── scripts/
│   ├── train.py              # QLoRA fine-tuning
│   ├── quantize.py           # AWQ quantization
│   ├── run_eval.py           # Evaluation runner
│   └── smoke_eval.py         # CI smoke test
├── eval/results/             # Benchmark outputs
│   ├── base/metrics.json
│   ├── finetuned/metrics.json
│   ├── quantized/metrics.json
│   ├── benchmark_results.json
│   └── error_analysis.json
├── docs/                     # Design docs and prep materials
├── tests/                    # Unit tests
├── notebooks/                # Kaggle evaluation notebooks
├── Dockerfile                # CUDA-based inference image
├── .github/workflows/ci.yml  # CI pipeline
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## Features

- Benchmarking for zero-shot, fine-tuned, and quantized models
- Automated evaluation pipeline for structured NLU predictions
- JSON output validation and metric computation
- Error analysis across success/failure categories
- CI pipeline for smoke testing and result verification
- Reproducible training and inference workflow
- Support for low-cost deployment via AWQ quantization

---

## Quick Start

### 1) Clone the repository

```bash
git clone https://github.com/najahaja/ft-bench.git
cd ft-bench
```

### 2) Install dependencies

```bash
pip install -e ".[dev]"
```

### 3) Run tests

```bash
pytest tests/ -v
```

### 4) Run benchmark evaluation

```bash
python scripts/run_eval.py --system base --output-dir eval/results/base
python scripts/run_eval.py --system finetuned --output-dir eval/results/finetuned
python scripts/run_eval.py --system quantized --output-dir eval/results/quantized
```

### 5) Run smoke test

```bash
python scripts/smoke_eval.py --n-samples 5 --system base
```

---

## Error Analysis

| Quadrant | Count | % |
|---|---:|---:|
| ✅ Both correct | ~68 | ~2.3% |
| 🎯 Fine-tuning fixed it | ~2,064 | ~69.4% |
| 💥 Fine-tuning broke it | ~10 | ~0.3% |
| ❌ Both wrong | ~832 | ~28.0% |

Fine-tuning corrected ~69.4% of previously wrong predictions with negligible regression.

---

## Cost and Efficiency

| System | Inference Cost | Relative |
|---|---:|---:|
| Base (fp16, T4) | ~$0.40/hr | 1× |
| Fine-Tuned (fp16, T4) | ~$0.40/hr | 1× |
| AWQ (int4, T4) | ~$0.24/hr | 0.6× |

AWQ reduces inference cost by roughly 40% while preserving accuracy.

---

## Hugging Face Models

| Model | Link |
|---|---|
| Fine-Tuned (QLoRA, fp16) | [najahaja/ftbench-qlora-llama3.2-3b](https://huggingface.co/najahaja/ftbench-qlora-llama3.2-3b) |
| LoRA Adapter | [najahaja/ftbench-qlora-llama3.2-3b-adapter](https://huggingface.co/najahaja/ftbench-qlora-llama3.2-3b-adapter) |
| AWQ (4-bit) | [najahaja/ftbench-qlora-llama3.2-3b-awq](https://huggingface.co/najahaja/ftbench-qlora-llama3.2-3b-awq) |

---

## Methodology

### Dataset: MASSIVE en-US

- Source: Amazon Science MASSIVE dataset
- Task: generate JSON with intent and slot values from user utterances
- Split: 11,481 train / 2,033 val / 2,974 test
- Intents: 60
- Slot types: 55

### System A: Zero-Shot Baseline

- Model: meta-llama/Llama-3.2-3B-Instruct (FP16)
- No fine-tuning, only instruction prompting
- Inference: batch_size=8, max_new_tokens=128, greedy decoding

### System B: QLoRA Fine-Tuning

- QLoRA: 4-bit NF4 base + FP16 LoRA adapters
- LoRA config: rank=16, alpha=32, dropout=0.05, all linear layers
- Training: 3 epochs, learning rate=2e-4, effective batch size=16
- Hardware: Kaggle T4 GPU (15GB VRAM)

### System C: AWQ 4-bit Quantization

- Method: Activation-aware Weight Quantization (INT4, group_size=128)
- Tools: llmcompressor + compressed-tensors
- Calibration: first 100 training samples

---

## Validation and Quality Checks

- Unit tests: 34 passing tests covering parsing, metrics, and bootstrapping
- JSON validity: >99.5%
- Bootstrap confidence intervals for more reliable statistical evaluation
- Leakage check removed 33 training samples with fingerprint overlap
- Full error breakdown for base-to-fine-tuned and fine-tuned-to-AWQ comparisons

---

## License

Copyright © 2026 najahaja. All rights reserved.

---

## Connect

If you like this project, please give it a star ⭐

- GitHub: [@najahaja](https://github.com/najahaja)
- LinkedIn: [najahaja](https://www.linkedin.com/in/ahamednajah)

---

See [docs/interview_prep.md](docs/interview_prep.md) for deeper technical notes and resume-ready explanations.
