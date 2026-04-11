"""
Prepare the OpenWebText dataset for toy model pre-training.

Usage:
    python prepare_openwebtext.py                     # Llama tokenizer (default)
    python prepare_openwebtext.py --model_arch qwen   # Qwen tokenizer

Downloads OpenWebText, tokenizes, chunks into fixed-length (2048) blocks,
and saves to disk. Llama and Qwen use different tokenizers so datasets are
saved to separate directories.

NOTE: Llama uses huggyllama/llama-7b (vocab=32000) for historical reasons.
meta-llama/Llama-2-7b-hf has the same vocab and works identically.
Qwen uses Qwen/Qwen2-7B (vocab=151936).
"""

import os
import argparse
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

TOKENIZERS = {
    "llama": ("huggyllama/llama-7b", "./datasets/openwebtext-2048"),
    "qwen":  ("Qwen/Qwen2-7B",      "./datasets/qwen/openwebtext-2048"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model_arch", type=str, default="llama", choices=["llama", "qwen"])
args = parser.parse_args()

tokenizer_name, output_dir = TOKENIZERS[args.model_arch]

# Load dataset
dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True)
dataset = dataset["train"].train_test_split(test_size=0.0005, seed=42357, shuffle=True)
dataset["val"] = dataset.pop("test")

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
eos = tokenizer.eos_token_id

def tokenize(example):
    tokens = tokenizer(example["text"], truncation=False, add_special_tokens=False)
    tokens["input_ids"] = tokens["input_ids"] + [eos]
    tokens["attention_mask"] = tokens["attention_mask"] + [1]
    return tokens

tokenized = dataset.map(
    tokenize, remove_columns=["text"], batched=False, num_proc=16,
    desc=f"Tokenizing ({args.model_arch})",
)

# Chunk into 2048-token blocks
block_size = 2048

def chunk_across_examples(dataset, block_size):
    buffer, input_blocks = [], []
    for example in dataset:
        buffer.extend(example["input_ids"])
        while len(buffer) >= block_size:
            input_blocks.append(buffer[:block_size])
            buffer = buffer[block_size:]
    return Dataset.from_dict({
        "input_ids": input_blocks,
        "labels": [block.copy() for block in input_blocks],
    })

lm_dataset = DatasetDict({
    split: chunk_across_examples(tokenized[split], block_size=block_size)
    for split in tokenized.keys()
})

os.makedirs(output_dir, exist_ok=True)
lm_dataset.save_to_disk(output_dir)
print(f"Saved to {output_dir}: {lm_dataset}")
