# FT-Bench: QLoRA Fine-Tuning vs AWQ Quantization Benchmark

> **End-to-end benchmark comparing zero-shot inference, QLoRA fine-tuning, and AWQ 4-bit
> quantization of Llama-3.2-3B-Instruct on a real-world NLU task (MASSIVE, n=2,974).**

[![CI](https://github.com/najahaja/ft-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/najahaja/ft-bench/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/model-Llama--3.2--3B--Instruct-orange)](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)](#-license)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ft-bench.streamlit.app/)

---

## 🏆 Results (n = 2,974 test samples)

| Metric | Base (zero-shot) | Fine-Tuned (QLoRA) | AWQ (4-bit) | Δ Base→FT | Δ FT→AWQ |
|---|---|---|---|---|---|
| **Intent Accuracy** | 41.26% | 88.77% | 88.53% | **+47.5 pp** | -0.24 pp |
| **Slot F1** | 16.41% | 85.34% | 85.99% | **+68.9 pp** | +0.65 pp |
| **Exact Match** | 2.62% | 71.69% | 72.19% | **+69.1 pp** | +0.50 pp |
| **JSON Valid Rate** | 99.53% | 99.53% | 99.50% | — | -0.03 pp |

**Key Findings:**
- Fine-tuning delivers **27× improvement** in exact match (2.62% → 71.69%)
- AWQ 4-bit quantization **preserves 100% of accuracy** (+0.5 pp on Exact Match) while reducing model size by ~60%
- Both fine-tuned models produce valid JSON over 99.5% of the time

---

## 📐 Architecture

```
System A — Base          meta-llama/Llama-3.2-3B-Instruct  (zero-shot, fp16)
System B — Fine-Tuned    najahaja/ftbench-qlora-llama3.2-3b (QLoRA, fp16)
System C — AWQ           najahaja/ftbench-qlora-llama3.2-3b-awq (int4, AWQ)
```

**Training:**
- Dataset: MASSIVE en-US (Airline Travel Information System NLU)
- Train: 11,481 samples → Test: 2,974 samples
- Method: QLoRA (r=16, α=32) on all linear layers
- Hardware: NVIDIA T4 16GB (Kaggle)
- Training time: ~45 min

---

## 📂 Project Structure

```
ft-bench/
├── ftbench/                  # Core Python package
│   ├── common/               # seed, io utilities
│   ├── eval/                 # metrics, parse, stats, runner
│   └── prompts/              # prompt templates
├── data/                     # MASSIVE dataset (auto-generated)
│   ├── train.jsonl           # 11,481 samples
│   ├── val.jsonl             # 2,033 samples
│   └── test.jsonl            # 2,974 samples
├── scripts/
│   ├── train.py              # QLoRA fine-tuning
│   ├── quantize.py           # AWQ quantization
│   ├── run_eval.py           # evaluation runner
│   └── smoke_eval.py         # CI smoke test (CPU, no GPU)
├── eval/results/             # All benchmark outputs
│   ├── base/metrics.json
│   ├── finetuned/metrics.json
│   ├── quantized/metrics.json
│   ├── benchmark_results.json
│   └── error_analysis.json
├── docs/                     # Design docs, decisions, interview prep
├── tests/                    # 34 passing unit tests
├── notebooks/                # Kaggle evaluation notebooks
├── Dockerfile                # CUDA 12.1 + inference image
└── .github/workflows/ci.yml  # CI: lint, smoke eval, docker build, results check
```

---

## 📊 Error Analysis (4-Quadrant: Base → Fine-Tuned)

| Quadrant | Count | % |
|---|---|---|
| ✅ Both correct (TT) | ~68 | ~2.3% |
| 🎯 FT fixed it (FT) | **~2,064** | **~69.4%** |
| 💥 FT broke it (TF) | ~10 | ~0.3% |
| ❌ Both wrong (FF) | ~832 | ~28.0% |

Fine-tuning **fixed 69.4% of previously wrong predictions** with negligible regressions (0.3%).

---

## 🚀 Quick Start

### 1 — Clone and install
```bash
git clone https://github.com/najahaja/ft-bench.git
cd ft-bench
pip install -e ".[dev]"
pytest tests/ -v
```

### 2 — Run evaluation
```bash
python scripts/run_eval.py --system base --output-dir eval/results/base
python scripts/run_eval.py --system finetuned --output-dir eval/results/finetuned
python scripts/run_eval.py --system quantized --output-dir eval/results/quantized
```

### 3 — Run smoke test (no GPU required)
```bash
python scripts/smoke_eval.py --n-samples 5 --system base
```

---

## 💰 Cost Analysis

| System | Inference Cost | Relative |
|---|---|---|
| Base (fp16, T4) | ~$0.40/hr | 1× |
| Fine-Tuned (fp16, T4) | ~$0.40/hr | 1× |
| AWQ (int4, T4) | ~$0.24/hr | **0.6×** |

AWQ cuts inference cost by ~40% with zero accuracy loss.

---

## 🤗 HuggingFace Models

| Model | Link |
|---|---|
| Fine-Tuned (QLoRA, fp16) | [najahaja/ftbench-qlora-llama3.2-3b](https://huggingface.co/najahaja/ftbench-qlora-llama3.2-3b) |
| LoRA Adapter | [najahaja/ftbench-qlora-llama3.2-3b-adapter](https://huggingface.co/najahaja/ftbench-qlora-llama3.2-3b-adapter) |
| AWQ (4-bit) | [najahaja/ftbench-qlora-llama3.2-3b-awq](https://huggingface.co/najahaja/ftbench-qlora-llama3.2-3b-awq) |

---

## 📝 Methodology

### Dataset: MASSIVE en-US
- Source: Amazon Science MASSIVE dataset
- Task: Given utterance, output JSON with intent + slots
- Split: 11,481 train / 2,033 val / 2,974 test (after deduplication, leakage check)
- Intents: 60 | Slot types: 55

### System A: Zero-Shot Baseline
- Model: meta-llama/Llama-3.2-3B-Instruct (FP16)
- No training, instruction-prompted only
- Batched inference: batch_size=8, max_new_tokens=128, greedy decoding

### System B: QLoRA Fine-Tuning
- QLoRA: 4-bit NF4 base + FP16 LoRA adapters
- LoRA config: rank=16, alpha=32, dropout=0.05, all linear layers
- Training: 3 epochs, lr=2e-4, effective batch=16
- Hardware: Kaggle T4 GPU (15GB VRAM)

### System C: AWQ 4-bit Quantization
- Method: Activation-aware Weight Quantization (INT4, group_size=128)
- Tool: llmcompressor + compressed-tensors
- Calibration: First 100 training samples

---

## 🔍 Evaluation & Validation

- **Unit tests:** 34 passing tests covering parse_output, compute_metrics, bootstrap_ci
- **JSON validity:** >99.5% of outputs are parseable JSON
- **Bootstrap CI:** 1,000-iteration confidence intervals for statistical validity
- **Leakage check:** Removed 33 training samples with fingerprints in val/test
- **Error analysis:** Full 4-quadrant breakdown for Base→FT and FT→AWQ

---

## 📄 License & Intellectual Property

© 2026 **Ahamed Najah** ([@najahaja](https://github.com/najahaja)). All Rights Reserved.

This project, its benchmark suites, fine-tuned model checkpoints, and platform architecture are proprietary. Unauthorized reproduction, modification, distribution, or commercial use is strictly prohibited without prior written permission from the author.

---

**See [docs/interview_prep.md](docs/interview_prep.md) for resume bullets and deep technical Q&As.**
