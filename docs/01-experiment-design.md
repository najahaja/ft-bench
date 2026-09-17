# FT-Bench: Experiment Design (Frozen)

## §1 Systems Under Test

| ID | Description | Identifier |
|----|-------------|------------|
| A  | Zero-shot prompted base model (Llama 3.2 3B Instruct) | `base` |
| B  | QLoRA fine-tuned, merged to BF16 | `finetuned` |
| C  | Fine-tuned + AWQ 4-bit via vLLM (llm-compressor) | `awq` |

## §2 Dataset

- **Source:** `AmazonScience/massive`, English config `en-US`
- **Splits:** use the OFFICIAL train/validation/test splits as returned by
  `load_dataset("AmazonScience/massive", "en-US")` — do NOT re-split.
- **Approximate sizes (en-US):** train ~11,514 / validation ~2,033 / test ~2,974
- **License:** CC-BY 4.0

## §3 Leakage Control

Before writing train/val/test JSONL:
1. Normalize each utterance: lowercase, collapse whitespace.
2. Compute MD5 fingerprint per utterance.
3. Compare fingerprint sets across splits.
4. If any train utterance fingerprint appears in validation or test, drop it
   from train and log the count to `data/README.md`.
5. Log format in `data/README.md`:
   ```
   train: N rows | val: N rows | test: N rows
   leakage_removed_from_train: N rows
   ```

## §4 Task Formulation

- **Input:** raw utterance (string)
- **Output:** strictly valid JSON matching NLUParse schema:
  ```json
  {"intent": "<one of 60 MASSIVE intents>", "slots": {"<slot_type>": "<verbatim substring>", ...}}
  ```
- **Vocabulary:** loaded from `configs/label_vocab.json`, which is generated
  by `scripts/prepare_dataset.py` from the real dataset — never hand-written.
- **Slot values** must be verbatim substrings of the input utterance.

## §5 Prompt Format

Prompts are built exclusively via `tokenizer.apply_chat_template()`.
No hand-written template strings. System message:

```
You are an NLU parser. Given an utterance, output a JSON object with exactly
two keys:
  "intent": one label from the provided vocabulary
  "slots": object mapping slot type names to verbatim substrings of the input

Output only the raw JSON. No markdown fences, no explanation.
Vocabulary will be provided in the user turn.
```

User turn format:
```
Vocabulary:
Intents: <comma-separated list from label_vocab.json>
Slot types: <comma-separated list from label_vocab.json>

Utterance: <input text>
```

## §6 Metrics

All metrics computed by `ftbench.eval.metrics` — identical formula across all
three systems:

| Metric | Definition |
|--------|-----------|
| `json_valid_rate` | % outputs parseable as JSON |
| `schema_valid_rate` | % valid JSON passing NLUParse validation |
| `intent_accuracy` | exact match on intent field |
| `slot_f1` | micro-averaged token-F1 across all predicted/true slot values |
| `exact_match` | intent correct AND all slot types/values correct simultaneously |
| `latency_p50_ms` | median end-to-end latency in ms |
| `latency_p95_ms` | 95th-percentile latency |
| `throughput_tok_s` | output tokens per second |

Bootstrap CI: 95% CI on `intent_accuracy` with n=1,000 resamples.

## §7 Evaluation Protocol

- **Test set:** official MASSIVE test split (~2,974 examples).
- **CI golden subset:** 100 examples frozen as `eval/golden/golden_100.jsonl`
  for regression gate. Sampled stratified by intent.
- **No data leakage:** test split is never used during training or prompt tuning.

## §8 Hyperparameter Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LoRA rank | r=16 | r=8 degraded intent_accuracy on val; r=32 no gain +60% VRAM |
| LoRA target modules | all linear | attention-only LoRA hurt JSON format adherence |
| Learning rate | 2e-4 | cosine schedule; 5e-4 caused gradient instability |
| Quantization | AWQ via llm-compressor | lower perplexity on task calibration data vs GPTQ |
| Batch size | 4 x grad_accum=4 (eff. 16) | largest fitting T4 with 4-bit base + BF16 activations |

## §9 Win Conditions

- `slot_f1`(B) > `slot_f1`(A) with non-overlapping 95% CI → fine-tuning works.
- `slot_f1`(C) >= 0.95 * `slot_f1`(B) → quantization preserves quality.
- `latency_p95_ms`(C) < `latency_p95_ms`(A) at concurrency=8 → vLLM wins on latency.

## §10 Benchmark Protocol

- Concurrency sweep: [1, 2, 4, 8, 16] simultaneous clients.
- 3 repetitions per level, 20 warmup requests discarded.
- Reported metric: mean throughput_tok_s and latency_p95_ms per level.

## §11 Reproducibility Requirements

- All random seeds pinned to 42.
- `configs/label_vocab.json` committed to repo (generated once from real data,
  not re-generated on each run).
- `eval/golden/golden_100.jsonl` committed to repo.
- Training run logged to W&B; run ID recorded in `docs/decisions.md`.
