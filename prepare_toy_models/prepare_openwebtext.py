"""
Prepare the OpenWebText dataset for toy model pre-training.

Downloads OpenWebText, tokenizes with Llama tokenizer, chunks into
fixed-length (2048) blocks, and saves to disk.

NOTE: We use the huggyllama/llama-7b tokenizer (vocab_size=32000) for
historical reasons. meta-llama/Llama-2-7b-hf is the more standard choice
and has the same vocab_size=32000, so either works identically. If you
switch to a tokenizer with a different vocab size, update the LlamaConfig
in train.py accordingly.
"""

import os
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True)

# Train/val split
dataset = dataset["train"].train_test_split(test_size=0.0005, seed=42357, shuffle=True)
dataset["val"] = dataset.pop("test")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
eos = tokenizer.eos_token_id

def tokenize(example):
    tokens = tokenizer(example["text"], truncation=False, add_special_tokens=False)
    tokens["input_ids"] = tokens["input_ids"] + [eos]
    tokens["attention_mask"] = tokens["attention_mask"] + [1]
    return tokens

tokenized = dataset.map(
    tokenize,
    remove_columns=["text"],
    batched=False,
    num_proc=16,
    desc="Tokenizing dataset",
)

# Group tokens into fixed-length chunks
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

output_dir = "./datasets/openwebtext-2048"
os.makedirs(output_dir, exist_ok=True)
lm_dataset.save_to_disk(output_dir)
print(f"Saved to {output_dir}: {lm_dataset}")
