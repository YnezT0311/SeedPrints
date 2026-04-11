# Preparing Toy Models

Scripts to reproduce the 160M-parameter toy models used in Tables 1–4.

Pre-trained toy models are available on [HuggingFace](https://huggingface.co/YnezT/SeedPrints-toy-models) and will be auto-downloaded by `test_toy_models.py` if not found locally. Use these scripts only if you want to train from scratch.

## Tokenizer Note

All scripts use `huggyllama/llama-7b` tokenizer (vocab_size=32000). This must be consistent across data preparation, training, and finetuning. If you switch to a different tokenizer, update `prepare_openwebtext.py`, `train.py`, `finetune.py`, and the `LlamaConfig.vocab_size` accordingly.

## Steps

### 1. Prepare datasets

```bash
# OpenWebText (for pre-training, ~8M documents)
python prepare_openwebtext.py

# Code Stack (for continual training, Table 3)
python prepare_code_stack.py
```

### 2. Pre-train with different seeds

```bash
# Train 4 models with different init seeds (same data order)
for seed in 42 123 1000 2000; do
    python train.py --init_seed $seed --global_seed 42
done
```

This produces for each seed:
- `model/init-llama-160M-seed-{seed}` — randomly initialized model (Table 1)
- `model/llama-160M-openwebtext-seed-{seed}` — pre-trained model (Tables 2, 4)

### 3. Continual training (Table 3)

```bash
for seed in 123 1000; do
    python finetune.py --init_seed $seed --finetune_set TinyStoriesV2_cleaned
    python finetune.py --init_seed $seed --finetune_set code_stack
done
```

This produces:
- `model/llama-160M-finetune-TinyStoriesV2_cleaned-seed-{seed}`
- `model/llama-160M-finetune-code_stack-seed-{seed}`
