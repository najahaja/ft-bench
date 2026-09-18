# Architecture & Hyperparameter Decisions

## 2026-09-18 — Phase 5: Training Sequence Length Measurement

**Problem:**
Hyperparameter `max_seq_len` needed verification against the empirical token length distribution of the MASSIVE dataset to prevent silent target truncation during supervised fine-tuning.

**Measurement Methodology:**
- Dataset: `data/train.jsonl` (11,481 utterances)
- Tokenizer: `meta-llama/Llama-3.2-3B-Instruct`
- Prompt formatter: `ftbench.prompts.templates.build_training_prompt()` (using official chat template with system message, label vocabulary of 60 intents and 55 slot types, user utterance, and target assistant JSON)

**Measured Token Length Distribution:**
- Count: 11,481
- Min: 534 tokens
- Mean: 547.25 tokens
- P50: 546.0 tokens
- P90: 559.0 tokens
- P95: 564.0 tokens
- P99: 575.0 tokens
- Max: 631 tokens

**Decision:**
- Set `max_seq_len: 640` in `configs/training.yaml`.
- Initial estimate of 320 tokens would have truncated 100% of all training examples before the target JSON was reached (due to the ~490 token vocabulary block in the system/user turn).
- 640 tokens covers 100% of samples (max 631 tokens) with headroom.
- At `max_seq_len: 640`, 4-bit base weights (~2.0 GB) + LoRA adapter + activations under batch size 4 per device and gradient accumulation 4 require ~4.5–5.5 GB VRAM, fitting comfortably within the Kaggle Tesla T4 (15.8 GB) VRAM budget.

## 2026-09-18 — Baseline Quality Evaluation Latency Flag

- The Phase 4 zero-shot run latency (`13,827ms p50 / 5.3 tok/s`) was measured using an unbatched Hugging Face `pipeline()` sequentially across Kaggle T4x2.
- Marked as `"latency_source": "unbatched_hf_pipeline_quality_run_only"` in `eval/results/base/metrics.json`.
- Per `docs/01-experiment-design.md` §6 & §10, official serving latency and throughput benchmarks must be conducted using vLLM on a single T4 GPU with concurrency sweeps [1, 2, 4, 8, 16] in Phase 7.