# FT-Bench — Master Phase Checklist

Keep this file in `docs/` and check items off as you go. Where a phase is
already done, its checklist still matters — re-verify anything you're unsure
about before building on top of it, the way we did with the dataset revert.

Current status markers below reflect where the project stood at time of
writing. Update them as you progress.

---

## Phase 0 — Design & Documentation ✅ DONE
- [x] `docs/00-design.md` written and frozen (model, dataset, task, tools, budget)
- [x] `docs/01-experiment-design.md` written and frozen (prompt format, schema,
      splits, metrics, benchmark protocol, decision thresholds)
- [x] Both docs are byte-identical to the versions actually agreed on — spot
      check `01-experiment-design.md`'s line count and a few distinctive
      phrases ("bootstrap", "micro-F1") to confirm it wasn't paraphrased by
      an agent at some point
- [x] `docs/decisions.md` exists and has an entry for every deviation +
      correction so far (Bitext→MASSIVE revert, AutoAWQ→llm-compressor, etc.)
- [x] `docs/failures.md` exists and is being appended to as real failures
      happen (Failures 1–6 thoroughly documented with Problem/Root Cause/Fix/Result)

## Phase 1 — Repo Scaffold + Core Package ✅ DONE
- [x] `ftbench/` installs via `pip install -e .`
- [x] `pytest tests/ -v` — all green (34/34 tests passing as of latest verification)
- [x] No file in the repo references `TicketAnalysis`, `Bitext`, `autoawq`,
      `sentiment`, `priority` as operational code (negative test guards are fine)
- [x] `.gitignore` covers: `.venv/`, `__pycache__/`, `*.egg-info/`, `data/*.jsonl`,
      `*.db`, `.env`, `wandb/`
- [x] Git history is clean — no committed `.venv`, `__pycache__`, or raw data files
- [x] No API keys, tokens, or secrets anywhere in git history
      (`git log --all -p | grep -i "hf_\|wandb_v1_"` returns nothing)

## Phase 2 — Dataset Pipeline ✅ DONE (reverted to MASSIVE)
- [x] `scripts/prepare_dataset.py` loads `AmazonScience/massive`, `en-US`,
      official splits — not a custom split, not subsampled
- [x] `configs/label_vocab.json` has real counts: **60 intents, 55 slot types**
      (not a placeholder, not Bitext's category/sentiment/priority fields)
- [x] Leakage check ran and reported a count (`33 train rows removed` per
      your last log) — logged to `data/README.md`
- [x] `data/train.jsonl` (11,481), `data/val.jsonl` (2,033),
      `data/test.jsonl` (2,974) exist with real content — spot check
      `head -3 data/train.jsonl` looks like real utterances + real labels,
      not placeholder text
- [x] Prompt formatting in the JSONL uses `tokenizer.apply_chat_template()`
      output, not a hand-built string

## Phase 3 — Serving Gateway Skeleton + Local CI ✅ DONE (per your last report)
- [x] `serving/gateway/main.py`, `routing.py`, `db.py` exist and tests pass
- [x] Field naming is consistent — `ticket_text` was a known leftover from
      the Bitext task; confirm it was renamed to `utterance` or decide
      explicitly to leave it and note why in `decisions.md`
- [x] SQLite (`serving/requests.db`) logs: request ID, timestamp, model
      version, input/output token counts, latency, status
- [x] `tests/test_gateway.py` passes against the current (MASSIVE-based) schema

## Phase 4 — Zero-Shot Baseline, System A ✅ DONE
- [x] Evaluated on the **full official test set** (2,974 examples), not a subset
- [x] Bootstrap 95% CIs computed for **all** quality metrics — intent_accuracy: 40.92% [39.11, 42.70]
- [x] `json_valid_rate` (99.8%), `schema_valid_rate` (70.75%), `intent_accuracy` (40.92%),
      `slot_f1` (16.08%), `exact_match` (2.56%) all recorded in `eval/results/base/metrics.json`
- [x] Latency/throughput numbers from this run (13,827ms/5.3 tok/s) are
      explicitly labeled as a quality-run artifact, NOT the final benchmark
      figure — the real benchmark number comes from Phase 10, single-GPU, vLLM
- [ ] Few-shot variant run too (per §2 of the design doc), if you're using it
- [ ] Contamination probe run (base model's behavior when label vocab is
      withheld from the prompt) — documented, not necessarily alarming, just honest

## Phase 5 — QLoRA Fine-Tuning, System B ✅ DONE
- [x] `configs/training.yaml` has every hyperparameter explicit: rank=16, alpha=32,
      dropout=0.05, target_modules (q,k,v,o,gate,up,down), LR=2e-4, batch size=4,
      grad accumulation=4 (effective=16), 3 epochs, cosine scheduler, warmup=0.03,
      weight_decay=0.01, seed=42
- [x] `max_seq_len` was actually measured (640 tokens)
- [x] Model loads on **one GPU only** (`device_map={"": 0}`)
- [x] `report_to` handled explicitly in code
- [x] `prepare_model_for_kbit_training()` runs before `get_peft_model()`
- [x] `model.config.use_cache = False` during training
- [x] `gradient_checkpointing_kwargs={"use_reentrant": False}` set
- [x] `remove_unused_columns=False` set (labels are pre-built)
- [x] Training actually progresses past step 0 — smooth convergence across 3 epochs
- [x] Loss decreased over time without diverging
- [x] `trainer.save_model()` explicitly called after `trainer.train()` completed to `checkpoints/final`
- [x] Checkpoints and merged model pushed to HF Hub:
      - Adapter: `najahaja/ftbench-qlora-llama3.2-3b-adapter`
      - Merged FP16: `najahaja/ftbench-qlora-llama3.2-3b`
- [x] `results/phase5_qlora_results.md` published and verified
- [x] `docs/failures.md` updated with WandbCallback crash, device_map multi-GPU crash, and push cell fix

## Phase 6 — Fine-Tuned Evaluation, System B ✅ DONE
- [x] Evaluated on the **identical** test set used for System A — full 2,974
      examples, same prompt format, same generation params (temp=0, seed=42)
- [x] Evaluated using `scripts/run_eval.py --system finetuned` with `eval/results/finetuned/metrics.json`
- [x] Metric results recorded:
      - Intent Accuracy: 88.60% (95% CI: [87.42, 89.64]) vs Base 40.92% (+47.68pp)
      - Slot F1: 85.87% vs Base 16.08% (+69.79pp)
      - Exact Match: 70.95% vs Base 2.56% (+68.39pp)
      - JSON Valid: 100%, Schema Valid: 99.87%
- [x] Win condition checked against §8: Exact Match improvement (+68.39pp) is massively positive and excludes zero
- [ ] 4-quadrant error analysis done: correct→correct, incorrect→correct,
      correct→incorrect, incorrect→incorrect, with example outputs for each
- [x] Results distinct from and never overwrite the Phase 4 baseline file

## Phase 7 — AWQ Quantization, System C ✅ DONE
- [x] Uses `llm-compressor`, confirmed NOT `autoawq`/`AutoAWQ`
- [x] Calibration data sourced from the MASSIVE train split (512 samples)
- [x] Quantized model saved and pushed to HF Hub:
      - Hub Repo: `najahaja/ftbench-qlora-llama3.2-3b-awq` (verified HTTP 200, uses `compressed-tensors` format)
- [ ] Peak GPU memory measured for base FP16 vs AWQ INT4 — record during Phase 9/10

## Phase 8 — Quantized Evaluation, System C 🔄 IN PROGRESS
- [x] Evaluation pipeline prepared in `scripts/run_eval.py`
- [x] Runtime dependency fix identified & resolved (`pip install compressed-tensors`, Failure 6 in `docs/failures.md`)
- [x] Initial dry-run / smoke-test executed (5 samples in `eval/results/quantized/metrics.json`)
- [ ] Full evaluation run on **all 2,974 test samples** via GPU (`python scripts/run_eval.py --system awq`)
- [ ] Bootstrap 95% CIs computed for AWQ metrics
- [ ] Decision thresholds from §8 applied: verify quality retention vs System B fine-tuned model (e.g. drop <= 3pp)
- [ ] Results saved to `eval/results/quantized/metrics.json` and committed

## Phase 9 — vLLM Serving (all three systems) ⏳ PENDING
- [ ] vLLM installed and running on the T4 with `--dtype float16` (not bf16)
- [ ] Confirmed only **one** vLLM instance runs at a time during any
      quality/benchmark measurement — never A, B, C simultaneously while timing
- [ ] OpenAI-compatible endpoint responds correctly for each system in turn:
      - System A: `meta-llama/Llama-3.2-3B-Instruct`
      - System B: `najahaja/ftbench-qlora-llama3.2-3b`
      - System C: `najahaja/ftbench-qlora-llama3.2-3b-awq`
- [ ] FastAPI gateway routes `/generate?system={base|ft|awq}` correctly to
      whichever vLLM instance is currently up
- [ ] `/health`, `/models` endpoints work
- [ ] Gateway does not duplicate functionality vLLM's own `/metrics` already provides

## Phase 10 — Benchmark: Latency, Throughput, GPU Telemetry ⏳ PENDING
- [ ] Entire benchmark run happens in **one uninterrupted Kaggle session**,
      **one physical GPU**, no other GPU workload concurrent
- [ ] Concurrency sweep is exactly `[1, 2, 4, 8, 16]` (per frozen design —
      confirm this wasn't silently changed to `[1,5,10,25,50]` again)
- [ ] 20-request warm-up discarded before each timed run
- [ ] Each concurrency level run **3 times**; median + spread reported, not
      a single sample
- [ ] `benchmark_env.json` written: GPU name, driver, CUDA version, vLLM
      version, dtype, gpu_memory_utilization, max_model_len, git commit hash
- [ ] Locust run in headless mode against `localhost` — never through the
      Cloudflare tunnel
- [ ] p50, p95, p99 latency; requests/sec; tokens/sec recorded for all three systems
- [ ] Peak + mean GPU memory and utilization recorded via pynvml for all three
- [ ] Fixed 200-utterance input sample used (drawn from test set, distinct
      from the 100-example CI golden set)

## Phase 11 — Cost Analysis ⏳ PENDING
- [ ] Cost/1k-requests computed from **measured** Phase 10 throughput ×
      **published** T4 instance list price (not an actual bill — you didn't pay)
- [ ] Explicitly labeled in the README as a calculation, not an expenditure
- [ ] Compared across base / fine-tuned / quantized

## Phase 12 — Monitoring (Prometheus + Grafana) ⏳ PENDING
- [ ] Self-hosted locally via Docker — not a paid cloud tier
- [ ] Prometheus scrapes both the FastAPI gateway and vLLM's native `/metrics`
- [ ] Grafana dashboard shows: request count, error count, latency, token
      throughput, and (if you can pipe it in) GPU utilization/memory
- [ ] If Grafana feels like too much for the time you have left, this phase
      is explicitly OPTIONAL per your zero-cost plan — drop it without guilt,
      just note the decision in `docs/decisions.md`

## Phase 13 — Streamlit Dashboard ⏳ PENDING
- [ ] Input → side-by-side base/FT/AWQ comparison
- [ ] Reads real `eval/results/*/metrics.json`, not hardcoded placeholder numbers
- [ ] Shows latency, model version, quality metrics
- [ ] If mock data is still in there anywhere, it's clearly labeled and gets
      replaced once real results exist — never silently left in

## Phase 14 — Docker ⏳ PENDING
- [ ] `docker-compose.yml` runs the CPU services (FastAPI, SQLite, Prometheus,
      Grafana, Streamlit) fully, validated locally
- [ ] GPU Dockerfile is written and documented with the exact
      `docker run --gpus all ...` command, but explicitly marked
      **untested on GPU** in the README — Kaggle/Colab don't expose Docker,
      this is a known, stated limitation, not a hidden gap
- [ ] Docker **Engine** used (in WSL2), not Docker Desktop, for licensing reasons

## Phase 15 — CI/CD ⏳ PENDING (workflow written, needs to actually run green)
- [ ] `.github/workflows/ci.yml` installs `requirements.txt` only (CPU),
      never `requirements-dev.txt` (which has vllm/unsloth/llmcompressor)
- [ ] Runs `pytest tests/`
- [ ] Runs the frozen 100-example golden-set eval and compares against a
      committed threshold
- [ ] Fails the build if quality regresses beyond the threshold (§8 of design doc)
- [ ] Actually confirmed green on GitHub's Actions tab, not just "should work"

## Phase 16 — Documentation & Final Results ⏳ PENDING
- [ ] Central results table built from committed JSON artifacts, not
      hand-typed numbers — every number traceable to a script + file
- [ ] Results table near the top of the README
- [ ] **Limitations section** covers: T4-class hardware/fp16/no FlashAttention;
      3B model chosen for VRAM fit and its effect on the fine-tuning delta;
      MASSIVE contamination risk; unvalidated GPU Docker path
- [ ] **Cost & Resource Usage section**: total cost $0; platform; GPU type;
      number of sessions; approximate training time; storage used; free-tier
      limitations hit; how the design adapted. Include the explicit
      distinction: "$0 spent" vs. "compute has real economic value, obtained
      via free tiers, not created for free"
- [ ] `docs/decisions.md` is complete and reads as an honest history,
      including the Bitext detour and why it was reverted
- [ ] `docs/failures.md` covers every real failure hit so far (WandbCallback
      crash, device_map multi-GPU crash, adapter path mismatch, torch/CUDA
      downgrade issue, wiped Kaggle working dir, missing compressed-tensors) in Problem → Investigation → Root Cause → Fix → Result form
- [ ] Every phase's "reproduce this" command actually documented and correct

## Phase 17 — Interview Prep & Resume Bullets ⏳ PENDING
- [ ] Only written after real results exist — no placeholder numbers
- [ ] Resume bullets use actual measured metrics, not the "~85–90%" estimate
      that appeared as a mock placeholder earlier in the project
- [ ] Can explain, from memory, why MASSIVE over Bitext, why 3B over 8B, why
      llm-compressor over AutoAWQ, why single-GPU-only benchmarking matters,
      and what the AWQ result on Turing hardware actually showed
- [ ] Can walk through at least one real failure from `docs/failures.md`
      end-to-end without notes — this is often the best interview material
      in the whole project, since it demonstrates debugging skill, not just
      following a tutorial

---

## Where you are right now

**Done:** Phases 0–7 (Design, Package Scaffold, Dataset Pipeline, Serving Gateway Skeleton, Zero-Shot Baseline A, QLoRA Fine-Tuning B, Fine-Tuned Evaluation B, AWQ Quantization C).  
**In progress:** Phase 8 — Running the full 2,974 test-set evaluation for System C (AWQ Quantized Model).  
**Not started:** Phases 9–17 (vLLM Serving, Concurrency Benchmark, Cost Analysis, Prometheus/Grafana, Streamlit, Docker, GitHub Actions CI/CD, Documentation, and Interview Prep).

**Immediate Next Action:**  
Run `python scripts/run_eval.py --system awq` on Kaggle GPU across all 2,974 test examples (with `compressed-tensors` installed) to complete Phase 8 evaluation, verify retention against the §8 threshold, and commit `eval/results/quantized/metrics.json`.
