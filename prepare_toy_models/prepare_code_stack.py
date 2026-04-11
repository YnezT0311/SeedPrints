"""
Prepare the Code Stack dataset for toy model continual training (Table 3).

Downloads a Python subset of The Stack, tokenizes, chunks into 2048-token
blocks, and saves to disk.

NOTE: Uses the same tokenizer as train.py (huggyllama/llama-7b, vocab=32000).
"""

import os
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# Download subset
snapshot_download(
    repo_id="tyzhu/the-stack-py",
    repo_type="dataset",
    allow_patterns=["the-stack_0.jsonl", "the-stack_1.jsonl", "the-stack_2.jsonl"],
    local_dir="./datasets/code-stack-subset",
)

# Load
dataset = load_dataset(
    "json",
    data_files=[
        "./datasets/code-stack-subset/the-stack_0.jsonl",
        "./datasets/code-stack-subset/the-stack_1.jsonl",
        "./datasets/code-stack-subset/the-stack_2.jsonl",
    ],
    split="train",
)

dataset = dataset.train_test_split(test_size=0.0005, seed=42357, shuffle=True)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
eos = tokenizer.eos_token_id

def tokenize(example):
    tokens = tokenizer(example["content"], truncation=False, add_special_tokens=False)
    tokens["input_ids"] = tokens["input_ids"] + [eos]
    tokens["attention_mask"] = tokens["attention_mask"] + [1]
    return tokens

tokenized = dataset.map(
    tokenize,
    remove_columns=["content"],
    batched=False,
    num_proc=16,
    desc="Tokenizing dataset",
)

# Chunk into 2048-token blocks
block_size = 2048

def chunk_across_examples(dataset, block_size):
    buffer = []
    input_blocks = []
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

output_dir = "./datasets/code-stack"
os.makedirs(output_dir, exist_ok=True)
lm_dataset.save_to_disk(output_dir)
print(f"Saved to {output_dir}: {lm_dataset}")
