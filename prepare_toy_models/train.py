"""
Pre-train a 160M-parameter toy model on OpenWebText.

Usage:
    python train.py --init_seed 1000 --model_arch llama
    python train.py --init_seed 1000 --model_arch qwen

This creates a randomly initialized model (seed=init_seed), then trains it
on OpenWebText with a fixed data order (seed=global_seed). Two models with
different init_seeds but the same global_seed will see exactly the same data
in the same order, differing only in their initialization.

NOTE: We use the huggyllama/llama-7b tokenizer (vocab_size=32000) for
historical reasons. meta-llama/Llama-2-7b-hf is the more standard choice
and has the same vocab_size=32000, so either works identically. If you
switch to a tokenizer with a different vocab size, update the LlamaConfig
below and the prepare scripts accordingly.

Qwen uses the Qwen/Qwen2-7B tokenizer (vocab_size=151936) and requires
its own prepared dataset (tokenized with the Qwen tokenizer).
"""

import os
import random
import argparse
import numpy as np
import torch
import wandb
from datasets import load_from_disk
from transformers import (AutoTokenizer, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)


ARCH_CONFIG = {
    "llama": {
        "tokenizer": "huggyllama/llama-7b",
        "config_cls": "LlamaConfig",
        "model_cls": "LlamaForCausalLM",
        "config_kwargs": dict(
            hidden_size=768, intermediate_size=2048,
            num_attention_heads=12, num_hidden_layers=12,
            max_position_embeddings=2048,
        ),
        "dataset_subdir": "",
        "prefix": "llama-160M",
    },
    "qwen": {
        "tokenizer": "Qwen/Qwen2-7B",
        "config_cls": "Qwen2Config",
        "model_cls": "Qwen2ForCausalLM",
        "config_kwargs": dict(
            hidden_size=768, intermediate_size=3072,
            num_attention_heads=12, num_hidden_layers=12,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            tie_word_embeddings=False,
            use_cache=False,
        ),
        "dataset_subdir": "qwen",
        "prefix": "qwen-160M",
    },
}


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    arch = ARCH_CONFIG[args.model_arch]
    prefix = arch["prefix"]

    wandb.init(
        project="seedprints-toy",
        name=f"{prefix}-openwebtext-seed-{args.init_seed}",
        config={"init_seed": args.init_seed, "global_seed": args.global_seed,
                "model_arch": args.model_arch},
    )

    # === Step 1: Initialize model with init_seed ===
    torch.manual_seed(args.init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.init_seed)

    tokenizer = AutoTokenizer.from_pretrained(arch["tokenizer"])

    # Import the right config/model class
    if args.model_arch == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        ConfigCls, ModelCls = LlamaConfig, LlamaForCausalLM
    else:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        ConfigCls, ModelCls = Qwen2Config, Qwen2ForCausalLM

    config_kwargs = dict(arch["config_kwargs"])
    config_kwargs.update(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    config = ConfigCls(**config_kwargs)
    model = ModelCls(config)

    # Save init model
    init_dir = os.path.join(args.output_dir, f"init-{prefix}-seed-{args.init_seed}")
    os.makedirs(init_dir, exist_ok=True)
    model.save_pretrained(init_dir)
    print(f"Saved init model to {init_dir}")

    # === Step 2: Set global seed for training ===
    set_seed(args.global_seed)

    # Load dataset
    dataset_path = args.dataset_path
    if arch["dataset_subdir"]:
        dataset_path = os.path.join(os.path.dirname(dataset_path),
                                     arch["dataset_subdir"],
                                     os.path.basename(dataset_path))
    lm_dataset = load_from_disk(dataset_path)

    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f"./{prefix}-openwebtext-seed-{args.init_seed}",
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

    save_dir = os.path.join(args.output_dir, f"{prefix}-openwebtext-seed-{args.init_seed}")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    print(f"Saved trained model to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch", type=str, default="llama", choices=["llama", "qwen"])
    parser.add_argument("--init_seed", type=int, default=1000)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, default="./datasets/openwebtext-2048")
    parser.add_argument("--output_dir", type=str, default="./model")
    main(parser.parse_args())
