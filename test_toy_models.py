"""
test_toy_models.py - SeedPrint test for toy (160M-parameter) models.

Model name formats:
  init-{seed}                  → random init model
  openwebtext-{seed}           → pre-trained on OpenWebText
  TinyStoriesV2_cleaned-{seed} → continual-trained on TinyStories
  code_stack-{seed}            → continual-trained on code
  {ckpt_idx}-{seed}            → checkpoint at training step

Default: embedding input + coset + perdim-only.
"""

import os
import json
import logging
import argparse
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import seedprint
import utils


# ── Model Loading ────────────────────────────────────────────────────────────

def parse_model_name(name):
    """Parse 'dataset-seed' or 'ckpt-seed' into (type, dataset_or_ckpt, seed)."""
    parts = name.rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse model name: {name}")
    prefix, seed_str = parts
    seed = int(seed_str)
    if prefix == "init":
        return "init", "init", seed
    if prefix.isdigit():
        return "checkpoint", prefix, seed
    return "trained", prefix, seed


def prepare_model(model_arch, model_type, dataset_or_ckpt, seed, root_dir):
    """Load or create a toy model. Returns (model, model_path)."""
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

    if model_type == "init":
        model_path = os.path.join(root_dir, f"model/init-{model_arch}-seed-{seed}")
        if os.path.exists(model_path):
            logging.info(f"Loading init model (seed={seed}) from {model_path}")
            model = LlamaForCausalLM.from_pretrained(model_path)
        else:
            logging.info(f"Creating init model (seed={seed})")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
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
            model.save_pretrained(model_path)
    elif model_type == "trained":
        dataset = dataset_or_ckpt
        dir_name = f"{model_arch}-{dataset}" if dataset == "openwebtext" \
                   else f"{model_arch}-finetune-{dataset}"
        model_path = os.path.join(root_dir, f"model/{dir_name}-seed-{seed}")
        logging.info(f"Loading trained model from {model_path}")
        model = LlamaForCausalLM.from_pretrained(model_path)
    elif model_type == "checkpoint":
        ckpt_idx = dataset_or_ckpt
        model_path = os.path.join(root_dir, f"{model_arch}-openwebtext-seed-{seed}/checkpoint-{ckpt_idx}")
        logging.info(f"Loading checkpoint from {model_path}")
        model = LlamaForCausalLM.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.eval().cuda()
    return model, model_path


# ── Feature Caching ──────────────────────────────────────────────────────────

def get_or_compute_features(model, model_path, input_type, input_path, cache_suffix):
    """Load cached features or compute + cache them."""
    cache_path = os.path.join(model_path, cache_suffix)
    if os.path.exists(cache_path):
        logging.info(f"Loading cached features from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    logging.info(f"Computing features for {model_path}")
    if input_type == "token":
        features = utils.get_hidden_states_from_tokens(model, input_path, batch_size=32)
    else:
        features = utils.get_hidden_states_from_embeddings(model, input_path, batch_size=32)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(features, cache_path)
    logging.info(f"Saved features to {cache_path}")
    return features


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SeedPrint test for toy (160M) models.")
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--model_arch", type=str, default="llama-160M",
                        choices=["llama-160M", "qwen-160M"])
    parser.add_argument("--input_type", type=str, default="embedding",
                        choices=["embedding", "token"],
                        help="Default: embedding (recommended for toy models).")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--fingerprint_len", type=int, default=1024)
    parser.add_argument("--buffer_k", type=int, default=seedprint.DEFAULT_BUFFER_K)
    parser.add_argument("--identity_mode", type=str, default="coset",
                        choices=["coset", "base"])
    parser.add_argument("--use_agg", action="store_true", default=False,
                        help="Include agg signal with Bonferroni correction.")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_dir = os.path.abspath(os.path.dirname(__file__))

    # Logging
    log_file = os.path.join(root_dir,
        f"results/{args.input_type}-base-{args.base_model}_target-{args.target_model}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    logging.info(f"Target: {args.target_model}, Base: {args.base_model}")
    logging.info(f"Input: {args.input_type}, Mode: {args.identity_mode}, "
                 f"Samples: {args.num_samples}, Len: {args.fingerprint_len}")

    # Parse model names
    target_type, target_data, target_seed = parse_model_name(args.target_model)
    base_type, base_data, base_seed = parse_model_name(args.base_model)

    # Prepare input
    if args.input_type == "token":
        input_path = utils.generate_random_tokens(
            os.path.join(script_dir, f"fingerprints/random_tokens_{args.num_samples}_{args.fingerprint_len}_vocab32000.pt"),
            args.num_samples, args.fingerprint_len)
        cache_suffix = f"fingerprint{args.num_samples}_hiddenstate_token_{args.fingerprint_len}.pt"
    else:
        # For embedding input, we need a model to estimate embedding statistics
        init_model_path = os.path.join(root_dir, f"model/init-{args.model_arch}-seed-{base_seed}")
        emb_path = os.path.join(script_dir,
            f"fingerprints/random_embeddings_{args.num_samples}_{args.fingerprint_len}.pt")
        if not os.path.exists(emb_path):
            init_model = LlamaForCausalLM.from_pretrained(init_model_path)
            utils.generate_random_embeddings(init_model, emb_path,
                                             args.num_samples, args.fingerprint_len)
            del init_model
        input_path = emb_path
        cache_suffix = f"fingerprint{args.num_samples}_hiddenstate_{args.input_type}_{args.fingerprint_len}.pt"

    # Load models and compute features
    target_model, target_path = prepare_model(
        args.model_arch, target_type, target_data, target_seed, root_dir)
    hs_target = get_or_compute_features(
        target_model, target_path, args.input_type, input_path, cache_suffix)
    del target_model; torch.cuda.empty_cache()

    base_model, base_path = prepare_model(
        args.model_arch, base_type, base_data, base_seed, root_dir)
    hs_base = get_or_compute_features(
        base_model, base_path, args.input_type, input_path, cache_suffix)
    del base_model; torch.cuda.empty_cache()

    # Run test
    logging.info(f"Running SeedPrint: target={args.target_model} vs base={args.base_model}")
    logging.info(f"Shapes: target={hs_target.shape}, base={hs_base.shape}")

    results = seedprint.run_test(
        hs_base, hs_target,
        buffer_k=args.buffer_k,
        identity_mode=args.identity_mode,
        use_agg=args.use_agg,
    )

    logging.info(f"z_perdim: {results['z_perdim']:.4f}")
    if args.use_agg:
        logging.info(f"z_agg: {results['z_agg']:.4f}")
    logging.info(f"p_value: {results['p_value']:.4g}")
    logging.info(f"k: {results['k']}")

    # Save JSON
    json_file = os.path.join(root_dir,
        f"results/json/{args.input_type}-base-{args.base_model}_target-{args.target_model}.json")
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {json_file}")


if __name__ == "__main__":
    main()
