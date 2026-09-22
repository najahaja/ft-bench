# Phase 5 — QLoRA Fine-Tuning Results

## Model

- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Fine-tuning method: QLoRA
- Quantization during training: 4-bit NF4
- Compute dtype: FP16
- Maximum sequence length: 640

## LoRA Configuration

- Rank (`r`): 16
- Alpha: 32
- Dropout: 0.05
- Target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

## Training Configuration

- Epochs: 3
- Learning rate: `2e-4`
- Per-device train batch size: 4
- Gradient accumulation steps: 4
- Effective batch size: 16
- Scheduler: cosine
- Warmup ratio: 0.03
- Weight decay: 0.01
- Optimizer: `paged_adamw_8bit`
- Gradient checkpointing: enabled
- Seed: 42

## Artifacts

### LoRA Adapter

Hugging Face:

`najahaja/ftbench-qlora-llama3.2-3b-adapter`

### Merged FP16 Model

Hugging Face:

`najahaja/ftbench-qlora-llama3.2-3b`

The merged model consists of two SafeTensors shards and can be loaded directly with Transformers.

## Status

Phase 5 completed successfully.

The trained adapter and merged FP16 model were successfully uploaded to Hugging Face.
