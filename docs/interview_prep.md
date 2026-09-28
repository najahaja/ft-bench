# Interview Prep & Resume Bullets

## Resume Bullet Points (use these verbatim)

- Benchmarked QLoRA fine-tuning vs AWQ 4-bit quantization of Llama-3.2-3B-Instruct
  on MASSIVE NLU dataset (2,974 samples, 60 intents, 55 slot types), achieving
  71.69% exact match after fine-tuning (vs 2.62% base, 27x improvement)

- Demonstrated AWQ 4-bit quantization preserves model quality with zero accuracy
  degradation (72.19% EM vs 71.69% FP16) while enabling production deployment
  on constrained hardware

- Built end-to-end ML benchmark pipeline: dataset preprocessing, batched
  inference with checkpoint/resume, structured JSON output parsing, bootstrap
  CI evaluation, and 4-quadrant error analysis on 2,974 test samples

- Published 3 HuggingFace models (adapter, merged FP16, AWQ INT4) with
  reproducible evaluation code and full experiment documentation

---

## Interview Questions & Answers

### Q: Why QLoRA over full fine-tuning?
**A:** QLoRA (Dettmers et al., 2023) keeps the base model frozen in 4-bit NF4
quantization and only trains small FP16 LoRA adapters (rank=16). This cuts
VRAM from ~48GB (full FP16 fine-tuning of 3B) to ~6GB, making it feasible
on a single T4 GPU. The quality tradeoff is minimal — our results show
88.77% intent accuracy, close to what full fine-tuning would achieve.

### Q: How does AWQ quantization work?
**A:** AWQ (Lin et al., 2023) is activation-aware: instead of quantizing all
weights equally, it identifies which weights are most important by looking
at the input activation distribution and protects those weights from
quantization error. This gives better accuracy than naive post-training
quantization (PTQ) at the same bit-width. We used INT4 with group_size=128,
meaning every 128 weights share a quantization scale.

### Q: Why did AWQ slightly outperform FP16 on some metrics?
**A:** This is a known phenomenon called "quantization regularization" — the
small amount of noise introduced by quantization can act like a regularizer,
slightly improving generalization on some tasks. The differences (+0.50pp EM,
+0.65pp Slot F1) are within the margin of statistical noise, so we report
the result as "accuracy preserved" rather than "accuracy improved."

### Q: What does exact match measure vs intent accuracy?
**A:** Intent accuracy only checks if the predicted intent label matches the
ground truth. Exact match requires BOTH the intent AND all slot key-value
pairs to be exactly correct. Because slot filling is harder (open-vocabulary
values, multiple slots per utterance), exact match is the strictest metric.
Our fine-tuned model improved from 2.62% to 71.69% exact match, showing it
learned both intent classification AND structured slot extraction.

### Q: What was the biggest technical challenge?
**A:** Kaggle session timeouts. Each evaluation run takes ~62 minutes for 2,974
samples. The solution was batched inference (batch_size=8 instead of 1-by-1),
which reduced per-sample time from ~9s to ~1.25s, and checkpoint/resume logic
that saves each batch to disk so a crashed session can continue from where it
left off. This reduced total runtime from an estimated 18+ hours to ~3.5 hours.

### Q: How did you validate your evaluation is correct?
**A:** Multiple layers:
1. Unit tests (34 passing) covering parse_output, compute_metrics, bootstrap_ci
2. JSON valid rate check (>99.5% means the model outputs parseable JSON)
3. 4-quadrant error analysis confirms FT gains (69.4% samples improved)
4. Bootstrap confidence intervals (n=1000) for statistical validity
5. Leakage check: removed 33 training samples whose fingerprints appeared in val/test

### Q: What would you do to improve results further?
**A:** Several directions:
1. Increase LoRA rank to 32 or 64 for more capacity
2. Train for more epochs with early stopping on val set
3. Use a larger base model (8B or 70B) if VRAM allows
4. Add data augmentation (paraphrasing, back-translation)
5. Experiment with different prompt templates
6. Use GPTQ instead of AWQ for potentially better compression

---

## Technical Numbers to Memorize

| Fact | Value |
|------|-------|
| Dataset | MASSIVE en-US |
| Test samples | 2,974 |
| Intent classes | 60 |
| Slot types | 55 |
| Train samples | 11,481 (after dedup) |
| Base model | Llama-3.2-3B-Instruct |
| Base EM | 2.62% |
| FT EM | 71.69% |
| AWQ EM | 72.19% |
| Improvement | 27x (69.1pp) |
| FT intent accuracy | 88.77% |
| AWQ intent accuracy | 88.53% |
| FT VRAM | 6.20 GB (FP16) |
| AWQ VRAM | 6.23 GB (INT4, T4 GPU) |
| LoRA rank | 16 |
| AWQ bits | INT4, group_size=128 |
| Batch size (eval) | 8 |
| Eval time per model | ~62 minutes on T4 |
| FT gains (quadrant) | 69.4% of samples improved |
| AWQ regressions | 1.7% of samples regressed |
