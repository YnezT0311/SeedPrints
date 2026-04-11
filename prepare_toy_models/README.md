# Preparing Toy Models

Scripts to reproduce the 160M-parameter toy models used in Tables 1–4. Supports both Llama and Qwen architectures via `--model_arch`.

Pre-trained toy models are available on [HuggingFace](https://huggingface.co/YnezT/SeedPrints-toy-models) and will be auto-downloaded by `test_toy_models.py` if not found locally. Use these scripts only if you want to train from scratch.

## Tokenizer Note

- **Llama**: uses `huggyllama/llama-7b` tokenizer (vocab_size=32000) for historical reasons. `meta-llama/Llama-2-7b-hf` has the same vocab_size=32000 and works identically.
- **Qwen**: uses `Qwen/Qwen2-7B` tokenizer (vocab_size=151936). Requires its own prepared dataset.

## Steps

### 1. Prepare datasets

```bash
# Llama: OpenWebText tokenized with Llama tokenizer
python prepare_openwebtext.py

# Qwen: OpenWebText tokenized with Qwen tokenizer
python prepare_openwebtext_qwen.py

# Code Stack (for continual training, Table 3)
python prepare_code_stack.py
```

### 2. Pre-train with different seeds

```bash
# Llama models
for seed in 42 123 1000 2000; do
    python train.py --model_arch llama --init_seed $seed --global_seed 42
done

# Qwen models
for seed in 42 123 1000 2000; do
    python train.py --model_arch qwen --init_seed $seed --global_seed 42
done
```

This produces for each seed:
- `model/init-{llama,qwen}-160M-seed-{seed}` — randomly initialized model (Table 1)
- `model/{llama,qwen}-160M-openwebtext-seed-{seed}` — pre-trained model (Tables 2, 4)

### 3. Continual training (Table 3)

```bash
for seed in 123 1000; do
    python finetune.py --model_arch llama --init_seed $seed --finetune_set TinyStoriesV2_cleaned
    python finetune.py --model_arch llama --init_seed $seed --finetune_set code_stack
done
```

This produces:
- `model/{llama,qwen}-160M-finetune-TinyStoriesV2_cleaned-seed-{seed}`
- `model/{llama,qwen}-160M-finetune-code_stack-seed-{seed}`
