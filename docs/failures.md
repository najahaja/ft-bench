# Failure Log

Every failure documented here is real — no polish, no hiding.
Format per entry: Problem → Investigation → Root Cause → Fix → Result.
This file is required by Phase 16 and is interview material (Phase 17).

---

## Failure 1 — WandbCallback crash on training start

**Phase:** 5 (QLoRA Fine-Tuning)  
**Date:** 2026-09-18

### Problem
Training crashed immediately after the first step with:
```
AttributeError: 'WandbCallback' in callbacks list failed during on_log:
  module 'wandb' has no attribute 'run' (or run is None)
```
The crash happened before any gradient steps were recorded. Zero training progress.

### Investigation
- `wandb.login()` returned `True` — the API key was valid and login succeeded.
- But `wandb.run` was `None` — no `wandb.init()` had been called yet.
- The HuggingFace `WandbCallback` calls `wandb.log()` inside `on_log`, which
  requires an active run object. If the run was never initialized, the callback
  crashes on the very first log event at step 1.
- Setting `report_to="wandb"` while having `WANDB_DISABLED=true` in the environment
  created a contradictory state: Trainer was told to use W&B, but the env var
  silently blocked `wandb.init()` from running.

### Root Cause
The deprecated `WANDB_DISABLED` environment variable was set from a previous notebook
cell that tried to "safely disable wandb". This prevented `wandb.init()` from firing
even though `report_to="wandb"` was explicitly set in `TrainingArguments`, leaving
`wandb.run = None` when the callback first tried to log.

`WANDB_DISABLED` has been deprecated since wandb 0.13. It no longer reliably
suppresses wandb — it only breaks it.

### Fix
1. Removed all use of `WANDB_DISABLED` env var.
2. Added a login-success gate in `train_qlora.py`:
   ```python
   use_wandb = False
   if wandb_key:
       try:
           wandb.login(key=wandb_key, relogin=True)
           use_wandb = True
       except Exception:
           use_wandb = False
   report_to = ["wandb"] if use_wandb else "none"
   ```
3. `report_to` is now set programmatically based on `use_wandb`, never hardcoded
   or overridden by env vars.
4. If W&B login fails for any reason, training continues with `report_to="none"` —
   never crashes.

### Result
Training starts cleanly. W&B run initializes before the first log event.
Fallback to `report_to="none"` works correctly when credentials are missing.

---

## Failure 2 — device_map="auto" split model across two T4 GPUs

**Phase:** 5 (QLoRA Fine-Tuning)  
**Date:** 2026-09-18

### Problem
Training appeared to start — progress bar advanced — but crashed after ~10 steps:
```
RuntimeError: Expected all tensors to be on the same device, but found
at least two devices, cuda:0 and cuda:1!
```

### Investigation
- `torch.cuda.device_count()` returned `2` on Kaggle T4×2.
- `device_map="auto"` uses HuggingFace accelerate's device placement heuristic,
  which saw two GPUs and split the model layers across both.
- `bitsandbytes` NF4 4-bit quantization requires all quantized layers to reside
  on a single device. The mixed-device placement caused a tensor device mismatch
  during the backward pass.
- The crash was not visible during model loading — only triggered at first backward.

### Root Cause
`device_map="auto"` is correct for multi-GPU inference but is incompatible with
`bitsandbytes` QLoRA training on Kaggle's T4×2 configuration. The Kaggle environment
silently exposes two GPU indices even when only one is needed.

### Fix
1. Replaced `device_map="auto"` with `device_map={"": 0}` — pins all model layers
   to GPU 0 unconditionally.
2. Added a tripwire assertion immediately after model loading:
   ```python
   devices = {p.device for p in model.parameters()}
   assert len(devices) == 1, f"Multi-device model detected: {devices}"
   print(f"[OK] All parameters on single device: {devices}")
   ```
3. Added an inline comment in `configs/training.yaml` explaining why `"auto"` is banned
   for this training configuration.

### Result
Training runs entirely on GPU 0. The tripwire assertion prints `[OK]` at every
session startup, confirming single-device placement. Crash eliminated.

---

## Failure 3 — Adapter path mismatch; merge step failed silently

**Phase:** 5 (merge step after training)  
**Date:** 2026-09-19

### Problem
After training completed successfully, `merge_adapter.py` ran without error but
produced an empty `models/finetuned/` directory. The HF Hub push step then failed:
```
huggingface_hub.errors.HFValidationError: Repo id must be in the form
'repo_name' or 'namespace/repo_name': '/kaggle/working/models/finetuned'
```
The error message was misleading — it looked like a path format issue, not a
missing-file issue.

### Investigation
- `trainer.save_model()` was being called automatically by HuggingFace Trainer
  after each epoch, saving to a numbered subdirectory like `checkpoints/checkpoint-858/`.
- The explicit `trainer.model.save_pretrained()` call was targeting a different path.
- `merge_adapter.py` used a hardcoded path (`models/finetuned/`) that contained
  no `adapter_config.json`.
- PEFT's `from_pretrained()` receiving an empty directory has no way to distinguish
  "empty local path" from "malformed HF repo ID" — it falls through to the HF Hub
  lookup and raises `HFValidationError`.

### Root Cause
Two separate save paths (Trainer's automatic epoch saves vs explicit `save_pretrained`)
writing to different locations. The merge script assumed a fixed path that matched neither.

### Fix
1. Added three explicit `save_pretrained()` calls in `train_qlora.py` after
   `trainer.train()` completes — to `final_adapter/`, `output_dir/`, and `final/`.
2. The merge function now takes `adapter_path` as an explicit argument.
3. Added a file existence assertion before calling merge:
   ```python
   assert os.path.exists(os.path.join(adapter_path, "adapter_config.json")), \
       f"No adapter found at {adapter_path} — check save_pretrained paths"
   ```

### Result
Merge step reliably produces `models/finetuned/` with a full merged FP16 model.
Both adapter (`najahaja/ftbench-qlora-llama3.2-3b-adapter`) and merged model
(`najahaja/ftbench-qlora-llama3.2-3b`) confirmed on HF Hub.

---

## Failure 4 — torch/CUDA version downgrade broke bitsandbytes

**Phase:** 5 (Environment setup)  
**Date:** 2026-09-17

### Problem
After running `pip install unsloth[colab-new]`, bitsandbytes failed at import:
```
CUDA Setup failed despite GPU being available. Please run the following
command to get more information: python -m bitsandbytes
bitsandbytes/cuda_setup/main.py: multiple CUDA environment variables set
```

### Investigation
- `torch.__version__` returned `2.0.1+cu118` after the install — it had been
  downgraded from `2.3.1+cu121`.
- The Kaggle T4 driver supports CUDA 12.1. The `cu118` torch wheel expects CUDA
  11.8 libraries, which are not the primary CUDA libraries on Kaggle.
- `bitsandbytes` selects its CUDA backend at import time and failed to find a
  compatible library under the cu118 build.

### Root Cause
`unsloth[colab-new]`'s extras are designed for Google Colab T4 (which ships CUDA 11.8),
not Kaggle (which ships CUDA 12.1). The pip resolver silently downgraded torch to
satisfy unsloth's pinned dependency. The downgrade was not flagged as an error.

### Fix
Force-reinstall torch for CUDA 12.1 immediately after installing unsloth:
```bash
pip install torch==2.3.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --force-reinstall -q
pip install bitsandbytes --upgrade -q
```
Critical: torch must be pinned **after** unsloth, not before — pip will silently
downgrade it if unsloth runs last.

### Result
`torch.__version__` = `2.3.1+cu121`. bitsandbytes imports cleanly. Confirmed with
`python -c "import bitsandbytes as bnb; print(bnb.__version__)"`.

---

## Failure 5 — Phase 6 and 7 outputs lost due to Kaggle session wipe

**Phase:** 6 (Fine-Tuned Eval) and 7 (AWQ Quantization)  
**Date:** 2026-09-23

### Problem
Phase 6 evaluation and Phase 7 AWQ quantization were run and appeared to complete
successfully inside a Kaggle session. But when the session ended, all outputs in
`/kaggle/working/` were discarded. None were committed to GitHub or pushed to HF Hub:

- `eval/results/finetuned/metrics.json` — never committed, does not exist in git
- `eval/results/awq/metrics.json` — never committed, does not exist in git
- AWQ model — never pushed; HF Hub repo `najahaja/ftbench-qlora-llama3.2-3b-awq` does not exist
- No Phase 6 or 7 commits visible in `git log --all --oneline` on `origin/main`

### Investigation
- `git log --all --oneline` confirms the most recent commit is `79bc367`
  ("docs: add Phase 5 QLoRA training results").
- `eval/results/finetuned/` and `eval/results/awq/` directories do not exist
  in the local working tree or in any remote branch.
- HF Hub check for `najahaja/ftbench-qlora-llama3.2-3b-awq` returns 404.

### Root Cause
The notebook cells that ran Phase 6 and 7 did not include a final step to push
results. When the Kaggle session ended normally, all of `/kaggle/working/` was wiped.
This is standard Kaggle behavior — it is not a bug — but the notebook must explicitly
account for it.

### Fix
Every phase notebook now ends with a mandatory push cell that:
1. Commits all new `eval/results/*/metrics.json` files via `git commit && git push`.
2. Pushes any model artifacts to HF Hub via `.push_to_hub()`.
3. Both steps run unconditionally — not wrapped in `try/except` that swallows failures.
4. Assertions confirm the push succeeded before the cell is marked done.

Phase 6 and 7 must be re-run from scratch in a new Kaggle session.

### Result
Fix documented and applied going forward.
`phase_diagnostic.py` now detects this condition explicitly and reports it in the
phase summary table with a clear "re-run required" message.

---

---

## Failure 6 — Missing compressed-tensors dependency during AWQ evaluation

**Phase:** 8 (AWQ Quantized Evaluation)  
**Date:** 2026-09-23

### Problem
When evaluating the AWQ quantized model with `scripts/run_eval.py --system awq`, the script crashed immediately with:
```
ImportError: compressed_tensors is not installed and is required for compressed-tensors quantization. Please install it with `pip install compressed-tensors`.
```

### Investigation
- Inspecting `config.json` on Hugging Face Hub for `najahaja/ftbench-qlora-llama3.2-3b-awq` confirmed:
  `"quant_method": "compressed-tensors"`
- The model was quantized using Neural Magic's `llm-compressor` (per project design requirements).
- Unlike legacy `autoawq` models, `llm-compressor` models are saved in the modern `compressed-tensors` format.
- Hugging Face `transformers` natively supports loading `compressed-tensors` models, but delegates the dequantization kernels to the `compressed_tensors` Python package.
- The evaluation environment only installed `transformers`, `accelerate`, `bitsandbytes`, but missed `compressed-tensors`.

### Root Cause
Missing runtime dependency `compressed-tensors` required by `transformers` when loading `llm-compressor` quantized checkpoints.

### Fix
1. Added `compressed-tensors` and `llmcompressor` to setup dependencies.
2. In the Kaggle evaluation notebook, added `pip install -q compressed-tensors` before loading the model.

### Result
Model loads seamlessly via Hugging Face `pipeline` and `AutoModelForCausalLM`. Evaluation runs successfully.
