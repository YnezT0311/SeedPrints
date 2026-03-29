"""
test_foundation_models.py - SeedPrint test for foundation models.

Loads models from HuggingFace by tag, computes hidden states, caches them,
and runs the SeedPrint hypothesis test.

Default: token input + coset + perdim-only.
"""

import os
import json
import logging
import argparse
import torch

import seedprint
import utils
from model_config import (FOUNDATION_MODELS, DIM_INIT_FN, HF_TOKEN,
                          OLMO_BASE_MODEL, OLMO_CHECKPOINTS)


# ── Model Loading ────────────────────────────────────────────────────────────

def load_foundation_model(model_tag):
    """Load a foundation model by tag using utils.load_model."""
    if model_tag not in FOUNDATION_MODELS:
        raise ValueError(f"Unknown model tag: {model_tag}")
    model_name = FOUNDATION_MODELS[model_tag]
    hf_token = HF_TOKEN if "llama" in model_name.lower() else None
    logging.info(f"Loading model: {model_tag} ({model_name})")
    return utils.load_model(model_name, hf_token=hf_token)


def is_olmo_checkpoint(tag):
    return tag in OLMO_CHECKPOINTS


def load_olmo_model(checkpoint_name):
    """Load an OLMo-2-7B checkpoint (fp32 + sdpa)."""
    from transformers import AutoModelForCausalLM
    logging.info(f"Loading OLMo: {OLMO_BASE_MODEL} rev={checkpoint_name}")
    model = AutoModelForCausalLM.from_pretrained(
        OLMO_BASE_MODEL, revision=checkpoint_name,
        trust_remote_code=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    return model


# ── Feature Caching ──────────────────────────────────────────────────────────

def _cache_path(root, model_tag, num_samples, fingerprint_len, input_type, is_olmo=False):
    """Build cache file path."""
    if is_olmo:
        if input_type == "token":
            return os.path.join(root, f"{model_tag}_fingerprint{num_samples}_hiddenstate_token_{fingerprint_len}.pt")
        return os.path.join(root, f"{model_tag}_fingerprint{num_samples}_hiddenstate_{fingerprint_len}.pt")
    model_dir = os.path.join(root, f"model/foundations/{model_tag}")
    os.makedirs(model_dir, exist_ok=True)
    if input_type == "token":
        return os.path.join(model_dir, f"fingerprint{num_samples}_hiddenstate_token_{fingerprint_len}.pt")
    return os.path.join(model_dir, f"fingerprint{num_samples}_hiddenstate_embedding_{fingerprint_len}.pt")


def get_or_compute(model_loader, model_tag, input_type, input_path,
                   root, num_samples, fingerprint_len, is_olmo=False):
    """Load cached features, or load model → compute → cache → unload."""
    cp = _cache_path(root, model_tag, num_samples, fingerprint_len, input_type, is_olmo)
    if os.path.exists(cp):
        logging.info(f"Loading cached features from {cp}")
        return torch.load(cp, map_location="cpu")

    model = model_loader()
    logging.info(f"Computing {input_type} hidden states for {model_tag}")
    if input_type == "token":
        features = utils.get_hidden_states_from_tokens(model, input_path, batch_size=8)
    else:
        features = utils.get_hidden_states_from_embeddings(model, input_path, batch_size=8)

    os.makedirs(os.path.dirname(cp), exist_ok=True)
    torch.save(features, cp)
    logging.info(f"Saved features to {cp}")
    del model; torch.cuda.empty_cache()
    return features


# ── Random Input Preparation ─────────────────────────────────────────────────

def prepare_token_input(fingerprint_len, num_samples, script_dir):
    path = os.path.join(script_dir,
        f"fingerprints/random_tokens_{num_samples}_{fingerprint_len}_vocab32000.pt")
    return utils.generate_random_tokens(path, num_samples, fingerprint_len)


def prepare_embedding_input(hidden_size, fingerprint_len, num_samples, script_dir):
    path = os.path.join(script_dir,
        f"fingerprints/random_embeddings_{num_samples}_{fingerprint_len}_{hidden_size}.pt")
    if os.path.exists(path):
        return path
    if hidden_size not in DIM_INIT_FN:
        raise ValueError(f"No init factory for hidden_size={hidden_size}. "
                         f"Add one to model_config.DIM_INIT_FN.")
    init_model = DIM_INIT_FN[hidden_size]().to(torch.float32)
    utils.generate_random_embeddings(init_model, path, num_samples, fingerprint_len)
    del init_model
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SeedPrint test for foundation models.")
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--input_type", type=str, default="token",
                        choices=["embedding", "token"],
                        help="Default: token (recommended for foundation models).")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--fingerprint_len", type=int, default=1024)
    parser.add_argument("--buffer_k", type=int, default=seedprint.DEFAULT_BUFFER_K)
    parser.add_argument("--identity_mode", type=str, default="coset",
                        choices=["coset", "base"])
    parser.add_argument("--use_agg", action="store_true", default=False,
                        help="Include agg signal with Bonferroni correction.")
    args = parser.parse_args()

    script_dir = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))

    # Ensure output directories exist
    os.makedirs(os.path.join(root, "results/foundations/json"), exist_ok=True)
    os.makedirs(os.path.join(root, "results/olmo/json"), exist_ok=True)
    os.makedirs(os.path.join(root, "model/foundations"), exist_ok=True)
    os.makedirs(os.path.join(script_dir, "fingerprints"), exist_ok=True)

    # Logging
    is_olmo = is_olmo_checkpoint(args.target_model) or is_olmo_checkpoint(args.base_model)
    subdir = "olmo" if is_olmo else "foundations"
    log_file = os.path.join(root,
        f"results/{subdir}/{args.input_type}-base-{args.base_model}_target-{args.target_model}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    logging.info(f"Target: {args.target_model}, Base: {args.base_model}")
    logging.info(f"Input: {args.input_type}, Mode: {args.identity_mode}, "
                 f"Samples: {args.num_samples}")

    # Prepare random input
    if args.input_type == "token":
        input_path = prepare_token_input(
            args.fingerprint_len, args.num_samples, script_dir)
    else:
        # For embedding, we need to know hidden_size. Load config of one model.
        from transformers import AutoConfig
        tag = args.target_model if not is_olmo_checkpoint(args.target_model) else args.base_model
        if tag in FOUNDATION_MODELS:
            cfg = AutoConfig.from_pretrained(FOUNDATION_MODELS[tag], trust_remote_code=True)
            hidden_size = cfg.hidden_size
        else:
            hidden_size = 4096  # OLMo default
        input_path = prepare_embedding_input(
            hidden_size, args.fingerprint_len, args.num_samples, script_dir)

    # Model loaders (lazy)
    def _load_target():
        if is_olmo_checkpoint(args.target_model):
            return load_olmo_model(args.target_model)
        return load_foundation_model(args.target_model)

    def _load_base():
        if is_olmo_checkpoint(args.base_model):
            return load_olmo_model(args.base_model)
        return load_foundation_model(args.base_model)

    # Compute features
    hs_target = get_or_compute(
        _load_target, args.target_model, args.input_type, input_path,
        root, args.num_samples, args.fingerprint_len,
        is_olmo=is_olmo_checkpoint(args.target_model))

    hs_base = get_or_compute(
        _load_base, args.base_model, args.input_type, input_path,
        root, args.num_samples, args.fingerprint_len,
        is_olmo=is_olmo_checkpoint(args.base_model))

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
    json_file = os.path.join(root,
        f"results/{subdir}/json/{args.input_type}-base-{args.base_model}_target-{args.target_model}.json")
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {json_file}")


if __name__ == "__main__":
    main()
