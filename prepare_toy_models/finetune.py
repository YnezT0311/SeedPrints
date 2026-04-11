"""
Continual-train (finetune) a pre-trained 160M toy model on a new dataset.

Usage:
    python finetune.py --init_seed 1000 --model_arch llama --finetune_set TinyStoriesV2_cleaned
    python finetune.py --init_seed 1000 --model_arch qwen --finetune_set code_stack

This loads a model pre-trained on OpenWebText (from train.py) and continues
training on a different dataset. Used for Table 3 experiments.

NOTE: We use the huggyllama/llama-7b tokenizer (vocab_size=32000) for
historical reasons. meta-llama/Llama-2-7b-hf is the more standard choice
and has the same vocab_size=32000, so either works identically.

Qwen uses the Qwen/Qwen2-7B tokenizer (vocab_size=151936).
"""

import os
import random
import argparse
import numpy as np
import torch
import wandb
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import (AutoTokenizer, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)


ARCH_CONFIG = {
    "llama": {
        "tokenizer": "huggyllama/llama-7b",
        "model_cls": "LlamaForCausalLM",
        "prefix": "llama-160M",
        "dataset_subdir": "",
    },
    "qwen": {
        "tokenizer": "Qwen/Qwen2-7B",
        "model_cls": "Qwen2ForCausalLM",
        "prefix": "qwen-160M",
        "dataset_subdir": "qwen",
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
        name=f"{prefix}-finetune-{args.finetune_set}-seed-{args.init_seed}",
        config={"init_seed": args.init_seed, "global_seed": args.global_seed,
                "model_arch": args.model_arch, "finetune_set": args.finetune_set},
    )

    tokenizer = AutoTokenizer.from_pretrained(arch["tokenizer"])
    tokenizer.pad_token = tokenizer.eos_token

    # Import the right model class
    if args.model_arch == "llama":
        from transformers import LlamaForCausalLM as ModelCls
    else:
        from transformers import Qwen2ForCausalLM as ModelCls

    # Load or prepare finetuning dataset
    dataset_dir = args.dataset_dir
    if arch["dataset_subdir"]:
        dataset_dir = os.path.join(dataset_dir, arch["dataset_subdir"])
    dataset_path = os.path.join(dataset_dir, args.finetune_set)

    if os.path.exists(dataset_path):
        tokenized_dataset = load_from_disk(dataset_path)
    else:
        if args.finetune_set == "TinyStoriesV2_cleaned":
            dataset = load_dataset("fhswf/TinyStoriesV2_cleaned", trust_remote_code=True)
            n = len(dataset["train"]) // 20
            dataset["train"] = dataset["train"].shuffle(seed=args.global_seed).select(range(n))
        elif args.finetune_set == "BabyLM":
            dataset = load_dataset("cambridge-climb/BabyLM", trust_remote_code=True)
        elif args.finetune_set == "code_stack":
            code_path = os.path.join(os.path.dirname(dataset_dir), "code-stack")
            if not os.path.exists(code_path):
                raise RuntimeError("Run prepare_code_stack.py first")
            dataset = load_from_disk(code_path)
            os.makedirs(dataset_path, exist_ok=True)
            dataset.save_to_disk(dataset_path)
            tokenized_dataset = dataset
        else:
            raise ValueError(f"Unknown finetune_set: {args.finetune_set}")

        if args.finetune_set != "code_stack":
            print(f"Dataset: {args.finetune_set}, size: {len(dataset['train'])}")

            def tokenize_fn(example):
                return tokenizer(example["text"], truncation=True,
                                 max_length=2048, padding="max_length")

            split = dataset["train"].train_test_split(test_size=0.01, seed=args.global_seed)
            split_dataset = DatasetDict({"train": split["train"], "val": split["test"]})
            tokenized_dataset = split_dataset.map(tokenize_fn, batched=True,
                                                   remove_columns=["text"])
            os.makedirs(dataset_path, exist_ok=True)
            tokenized_dataset.save_to_disk(dataset_path)

    # Load pre-trained model
    pretrained_path = os.path.join(args.model_dir, f"{prefix}-openwebtext-seed-{args.init_seed}")
    print(f"Loading pre-trained model from {pretrained_path}")
    model = ModelCls.from_pretrained(pretrained_path)

    set_seed(args.global_seed)

    training_args = TrainingArguments(
        output_dir=f"./{prefix}-finetune-{args.finetune_set}-seed-{args.init_seed}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        bf16=True,
        num_train_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        report_to="wandb",
        seed=args.global_seed,
        dataloader_num_workers=16,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("val", tokenized_dataset.get("test")),
        data_collator=collator,
    )

    trainer.train()

    save_dir = os.path.join(args.model_dir, f"{prefix}-finetune-{args.finetune_set}-seed-{args.init_seed}")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    print(f"Saved finetuned model to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch", type=str, default="llama", choices=["llama", "qwen"])
    parser.add_argument("--init_seed", type=int, default=1000)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--finetune_set", type=str, default="TinyStoriesV2_cleaned",
                        choices=["TinyStoriesV2_cleaned", "BabyLM", "code_stack"])
    parser.add_argument("--dataset_dir", type=str, default="./datasets/")
    parser.add_argument("--model_dir", type=str, default="./model")
    main(parser.parse_args())
