"""
Prepare the OpenWebText dataset for Qwen toy model pre-training.

Same as prepare_openwebtext.py but uses the Qwen tokenizer (vocab_size=151936).
Qwen and Llama datasets must be prepared separately because their tokenizers differ.
"""

import os
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True)

dataset = dataset["train"].train_test_split(test_size=0.0005, seed=42357, shuffle=True)
dataset["val"] = dataset.pop("test")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
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
    desc="Tokenizing dataset (Qwen)",
)

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

output_dir = "./datasets/qwen/openwebtext-2048"
os.makedirs(output_dir, exist_ok=True)
lm_dataset.save_to_disk(output_dir)
print(f"Saved to {output_dir}: {lm_dataset}")
