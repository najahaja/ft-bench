# FT-Bench — Master Phase Checklist
# Last synced: 2026-09-26  (from phase_diagnostic.py on Kaggle)
# Source of truth: actual files + HF Hub.
# Verify anytime: python3 scripts/phase_diagnostic.py

---

## Phase 0 — Design & Documentation  🔄 IN-PROGRESS (3/4)
- [x] docs/00-design.md written and frozen
- [x] docs/01-experiment-design.md written and frozen
- [x] docs/decisions.md exists
- [~] docs/failures.md — was 0 bytes in git (never committed).
      FIXED in commit cb79b89 (pushed 2026-09-26). Verify after next git pull.

## Phase 1 — Repo Scaffold + Core Package  ✅ DONE
- [x] ftbench importable
- [x] ftbench/ directory exists
- [x] pytest: 34 passed
- [x] .gitignore exists

## Phase 2 — Dataset Pipeline (MASSIVE)  ✅ DONE
- [x] data/train.jsonl: 11,481 rows
- [x] data/val.jsonl:    2,033 rows
- [x] data/test.jsonl:   2,974 rows
- [x] 60 intents, 55 slots in configs/label_vocab.json
- [x] Leakage check: 33 rows removed

## Phase 3 — Serving Gateway Skeleton  ✅ DONE
- [x] serving/gateway/main.py
- [x] serving/gateway/routing.py
- [x] serving/gateway/db.py
- [x] 34 pytest tests green

## Phase 4 — Zero-Shot Baseline, System A  🔄 IN-PROGRESS (3/4)
- [x] eval/results/base/metrics.json — n_samples=2974, intent_acc=40.92%, EM=2.56%
- [x] intent_accuracy_95ci present
- [x] exact_match=2.56% recorded
- [ ] eval/results/base/records.jsonl — MISSING (run_baseline.py never saved it).
      Needed for 4-quadrant error analysis in Phase 6/8.
      Fix → Cell A below (5 min, CPU ok)

## Phase 5 — QLoRA Fine-Tuning, System B  🔄 IN-PROGRESS (3/4)
- [x] HF Hub adapter:  najahaja/ftbench-qlora-llama3.2-3b-adapter  ✓
- [x] HF Hub merged:   najahaja/ftbench-qlora-llama3.2-3b           ✓
- [x] results/phase5_qlora_results.md committed
- [~] docs/failures.md — same fix as Phase 0 (cb79b89). Verify after pull.

## Phase 6 — Fine-Tuned Evaluation, System B  ❌ PENDING (0/5)
- [ ] eval/results/finetuned/metrics.json n_samples=2974
      (Local file has 70.95% EM — was never committed. NOW IN GIT via cb79b89.
       But records.jsonl was never generated at all.)
- [ ] exact_match > 50% confirmed
- [ ] Bootstrap CIs (intent, slot, EM)
- [ ] eval/results/finetuned/records.jsonl — MISSING. Must run eval on GPU.
- [ ] eval/results/error_analysis.json (4-quadrant) — depends on records.jsonl

## Phase 7 — AWQ Quantization, System C  🔄 IN-PROGRESS (1/2)
- [x] HF Hub AWQ: najahaja/ftbench-qlora-llama3.2-3b-awq  ✓
- [ ] eval/results/quantized/vram_stats.json — MISSING. Record during Phase 8 run.

## Phase 8 — Quantized Evaluation, System C  ❌ PENDING (0/6)
- [ ] eval/results/quantized/metrics.json n_samples=2974 (currently: 5-sample smoke test)
- [ ] eval/results/quantized/records.jsonl 2974 lines    (currently: 5 lines)
- [ ] Bootstrap CIs (exact_match_95ci)
- [ ] exact_match > 30%
- [ ] eval/results/quantized/win_condition.json (sec8 pass/fail)
- [ ] eval/results/error_analysis.json (4-quadrant Base→FT and FT→AWQ)

## Phase 9 — vLLM Serving  ❌ PENDING
- [ ] serving/vllm/ scripts
- [ ] eval/results/benchmark_env.json

## Phase 10 — Benchmark: Latency / Throughput / GPU  ❌ PENDING
- [ ] eval/results/benchmark*.json
- [ ] load_testing/reports/*.csv (Locust)
- [ ] benchmark_env.json

## Phase 11 — Cost Analysis  ❌ PENDING
- [ ] eval/results/cost_comparison.json

## Phase 12 — Monitoring (Prometheus + Grafana)  ❌ PENDING
- [ ] monitoring/prometheus/*.yml
- [ ] monitoring/grafana/dashboards/*.json

## Phase 13 — Streamlit Dashboard  ❌ PENDING
- [ ] dashboard/*.py

## Phase 14 — Docker  ❌ PENDING
- [ ] docker-compose.yml
- [ ] Dockerfile

## Phase 15 — CI/CD (GitHub Actions)  ❌ PENDING
- [ ] .github/workflows/ci.yml

## Phase 16 — Documentation & Final Results  🔄 IN-PROGRESS (1/4)
- [ ] README.md (does not exist yet)
- [ ] README.md results table
- [x] docs/decisions.md
- [~] docs/failures.md (fixed in cb79b89 — verify)

## Phase 17 — Interview Prep & Resume Bullets  ❌ PENDING
- [ ] docs/interview.md or docs/resume_bullets.md
- [ ] Real AWQ 2974-sample metrics must exist first

---

## WHERE YOU ARE: Phase 6

**True current work:** Phase 6 is the first fully PENDING phase.
Before you can tick it done you also need a small fix in Phase 4
(base records.jsonl), which is Cell A below.

---

## EXACT NEXT STEPS (run in order)

### Cell A — Regenerate base records.jsonl  (5 min, GPU T4, after git pull)
Needed for Phase 4 completion AND 4-quadrant error analysis.
Run after Cell 1 (setup) completes and dataset is ready.

    import subprocess, sys, os
    os.chdir("/kaggle/working/ft-bench")

    # Re-run baseline eval — saves records.jsonl this time
    # Uses the same eval code (ftbench.eval.runner) — identical to original Phase 4
    subprocess.check_call([
        sys.executable, "scripts/run_eval.py",
        "--system", "base",
        "--output-dir", "eval/results/base",
    ])
    # This needs GPU because it loads meta-llama/Llama-3.2-3B-Instruct.
    # ~60-90 min on T4. If you want to avoid re-running the model,
    # see the mock alternative in Cell A2 below.

    # After it finishes:
    import subprocess
    subprocess.check_call(["git", "add", "eval/results/base/records.jsonl",
                           "eval/results/base/metrics.json"])
    subprocess.check_call(["git", "commit", "-m",
                           "results: Phase 4 base records.jsonl generated"])
    subprocess.check_call(["git", "push"])
    print("Phase 4 base records committed.")

> NOTE: If you cannot afford another 60-90 min for base, skip this for now
> and do it in the same session as Phase 6 eval. The 4-quadrant analysis
> (Cell 8.9) will just run base vs FT, which is fine.

---

### Cell B — Phase 6: Fine-Tuned Evaluation (60–90 min, GPU T4)
This is your main task. Generates finetuned/records.jsonl + full metrics.

    # Phase 6 | Cell B — Fine-Tuned Evaluation
    import os, time, torch
    from ftbench.common.seed import seed_everything
    from ftbench.common.io import read_jsonl
    from ftbench.eval.runner import run_eval
    from ftbench.eval.metrics import compute_metrics
    from ftbench.eval.stats import bootstrap_ci
    from ftbench.common.io import write_json
    from transformers import AutoTokenizer, AutoModelForCausalLM

    seed_everything(42)
    os.chdir("/kaggle/working/ft-bench")

    FT_MODEL_ID = "najahaja/ftbench-qlora-llama3.2-3b"
    OUTPUT_DIR  = "eval/results/finetuned"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(FT_MODEL_ID, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        FT_MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": 0},
        token=HF_TOKEN,
    )
    try:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
    except Exception:
        pass

    GEN_KWARGS = {
        "max_new_tokens": 256, "temperature": 0.1, "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    def generate_fn(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KWARGS)
        return tokenizer.decode(out[0][plen:], skip_special_tokens=True)

    test_samples = read_jsonl("data/test.jsonl")
    print(f"[+] Evaluating {len(test_samples)} samples (system=finetuned)")
    t0 = time.time()
    records = run_eval(test_samples, generate_fn, system_name="finetuned",
                       output_path=f"{OUTPUT_DIR}/records.jsonl", verbose=True)
    print(f"[✓] Done in {(time.time()-t0)/60:.1f} min — {len(records)} records")

    # Metrics + CIs
    metrics   = compute_metrics(records)
    intent_ci = bootstrap_ci(records, "intent_accuracy", n_bootstrap=1000, seed=42)
    slot_ci   = bootstrap_ci(records, "slot_f1",         n_bootstrap=1000, seed=42)
    em_ci     = bootstrap_ci(records, "exact_match",     n_bootstrap=1000, seed=42)

    result = {
        "system": "finetuned", "n_samples": len(records), "metrics": metrics,
        "intent_accuracy_95ci": {"point": intent_ci[0], "lower": intent_ci[1], "upper": intent_ci[2]},
        "slot_f1_95ci":         {"point": slot_ci[0],   "lower": slot_ci[1],   "upper": slot_ci[2]},
        "exact_match_95ci":     {"point": em_ci[0],     "lower": em_ci[1],     "upper": em_ci[2]},
    }
    write_json(result, f"{OUTPUT_DIR}/metrics.json")
    print(f"exact_match={metrics['exact_match']}%  EM_CI=[{em_ci[1]}, {em_ci[2]}]")

    # Commit (MANDATORY — do not wrap in try/except)
    import subprocess
    subprocess.check_call(["git", "add", "eval/results/finetuned/records.jsonl",
                           "eval/results/finetuned/metrics.json"])
    subprocess.check_call(["git", "commit", "-m",
                           "results: Phase 6 finetuned full eval — 2974 samples with CIs"])
    subprocess.check_call(["git", "push"])
    print("[✓] Phase 6 pushed.")

---

### Cell C — Phase 8: AWQ Evaluation + VRAM + Error Analysis (60–90 min, GPU T4)
Run in the SAME Kaggle session immediately after Cell B (model still in memory or reload).
See kaggle_notebook_guide_phase8.md Cells 8.4–8.10 for the full code.
Key additions vs the Phase 6 cell:
  - Load AWQ model instead of FT model
  - Cell 8.5: measure vram_stats.json (Phase 7 headline number)
  - Cell 8.8: win_condition.json (sec8 decision gate)
  - Cell 8.9: error_analysis.json (4-quadrant, needs both FT and base records)

---

### Cell D — Run Diagnostic to Confirm
After committing Phase 6 + Phase 8:

    import os
    os.chdir("/kaggle/working/ft-bench")
    import subprocess, sys
    subprocess.check_call(["git", "pull"])
    exec(open("scripts/phase_diagnostic.py").read())

Expected result: P4=DONE, P6=DONE, P7=DONE, P8=DONE.
If so, you are ready to move to Phase 9 (vLLM serving).
