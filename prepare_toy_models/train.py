"""
Pre-train a 160M-parameter Llama model on OpenWebText.

Usage:
    python train.py --init_seed 1000 --global_seed 42

This creates a randomly initialized model (seed=init_seed), then trains it
on OpenWebText with a fixed data order (seed=global_seed). Two models with
different init_seeds but the same global_seed will see exactly the same data
in the same order, differing only in their initialization.

NOTE: The tokenizer (huggyllama/llama-7b, vocab=32000) must match the one
used in prepare_openwebtext.py. If you switch tokenizers, update both files
and the LlamaConfig below.
"""

import os
import random
import argparse
import numpy as np
import torch
import wandb
from datasets import load_from_disk
from transformers import (AutoTokenizer, LlamaConfig, LlamaForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    wandb.init(
        project="seedprints-toy",
        name=f"llama-160M-openwebtext-seed-{args.init_seed}",
        config={"init_seed": args.init_seed, "global_seed": args.global_seed},
    )

    # === Step 1: Initialize model with init_seed ===
    # This seed ONLY affects the random initialization of model weights.
    torch.manual_seed(args.init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.init_seed)

    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    config = LlamaConfig(
        hidden_size=768, intermediate_size=2048,
        num_attention_heads=12, num_hidden_layers=12,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_position_embeddings=2048,
    )
    model = LlamaForCausalLM(config)

    # Save init model (for Table 1-2 experiments)
    init_dir = os.path.join(args.output_dir, f"init-llama-160M-seed-{args.init_seed}")
    os.makedirs(init_dir, exist_ok=True)
    model.save_pretrained(init_dir)
    print(f"Saved init model to {init_dir}")

    # === Step 2: Set global seed for training (data shuffling, dropout) ===
    set_seed(args.global_seed)

    # Load dataset (prepared by prepare_openwebtext.py)
    lm_dataset = load_from_disk(args.dataset_path)

    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f"./llama-160M-openwebtext-seed-{args.init_seed}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        bf16=True,
        num_train_epochs=1,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=200,
        logging_steps=10,
        save_steps=2000,
        eval_steps=5000,
        report_to="wandb",
        seed=args.global_seed,
        dataloader_num_workers=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["val"],
        data_collator=collator,
    )

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

    trainer.train()

    save_dir = os.path.join(args.output_dir, f"llama-160M-openwebtext-seed-{args.init_seed}")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    print(f"Saved trained model to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_seed", type=int, default=1000)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, default="./datasets/openwebtext-2048")
    parser.add_argument("--output_dir", type=str, default="./model")
    main(parser.parse_args())
