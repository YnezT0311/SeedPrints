"""
utils.py - Inference utilities for SeedPrint.

Provides:
  - Random input generation (token sequences and continuous embeddings)
  - Hidden state / logit extraction from transformer models
  - Model loading helpers with dtype promotion and attention backend selection
"""

import os
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ── Random Input Generation ──────────────────────────────────────────────────

def generate_random_tokens(save_path, num_samples, seq_length, min_vocab=32000, seed=42):
    """
    Generate or load cached random token ID sequences.

    Token IDs are sampled uniformly from [0, min_vocab).  min_vocab should be
    the smallest vocabulary size across all models in the experiment so that
    every ID is valid for every model.

    Args:
        save_path: path to save/load the tensor.
        num_samples: number of sequences.
        seq_length: tokens per sequence.
        min_vocab: upper bound for token IDs (exclusive).
        seed: random seed for reproducibility.
    Returns:
        path to [num_samples, seq_length] int64 tensor on disk.
    """
    if os.path.exists(save_path):
        logging.info(f"Random tokens cached at {save_path}")
        return save_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.manual_seed(seed)
    tokens = torch.randint(0, min_vocab, (num_samples, seq_length), dtype=torch.long)
    torch.save(tokens, save_path)
    logging.info(f"Saved random tokens ({num_samples}×{seq_length}) to {save_path}")
    return save_path


def generate_random_embeddings(model, save_path, num_samples, seq_length):
    """
    Generate or load cached random continuous embeddings ~ N(μ, σ).

    μ and σ are estimated from the model's embedding layer weights.
    The resulting tensor bypasses the embedding layer during inference.

    Args:
        model: transformer model (only embedding layer stats are used).
        save_path: path to save/load the tensor.
        num_samples: number of sequences.
        seq_length: tokens per sequence.
    Returns:
        path to [num_samples, seq_length, hidden_size] tensor on disk.
    """
    if os.path.exists(save_path):
        logging.info(f"Random embeddings cached at {save_path}")
        return save_path

    hidden_size = model.config.hidden_size
    embed_weight = model.model.embed_tokens.weight
    mu = embed_weight.mean().item()
    sigma = embed_weight.std().item()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    embeddings = torch.normal(mu, sigma, size=(num_samples, seq_length, hidden_size))
    torch.save(embeddings, save_path)
    logging.info(f"Saved random embeddings ({num_samples}×{seq_length}×{hidden_size}) to {save_path}")
    return save_path


# ── Hidden State Extraction ──────────────────────────────────────────────────

def _get_inner_model(model):
    """Return the inner transformer (without lm_head) to save memory."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model          # Llama, Qwen2, Gemma, OLMo
    if hasattr(model, "transformer"):
        return model.transformer     # GPT-style
    return None


def _adaptive_batch_size(hidden_size, default_bs, mode="token"):
    """Adjust batch size based on hidden dimension to avoid OOM."""
    if hidden_size >= 4096:
        return min(default_bs, 8)
    if hidden_size <= 1024:
        return max(default_bs, 128) if mode == "embedding" else default_bs
    return default_bs


def get_hidden_states_from_tokens(model, token_path, batch_size=32):
    """
    Forward pass random token IDs through model → last hidden state.

    Each model applies its own embedding layer, making the resulting hidden
    states model-specific even for shared token sequences.

    Args:
        model: transformer model on GPU.
        token_path: path to [N, seq_len] int64 tensor.
        batch_size: inference batch size (auto-adjusted for large models).
    Returns:
        Tensor [N, hidden_size] (float32, CPU).
    """
    device = model.device
    tokens = torch.load(token_path, map_location="cpu", weights_only=True)
    hidden_size = getattr(model.config, "hidden_size", 0)
    batch_size = _adaptive_batch_size(hidden_size, batch_size, mode="token")
    inner = _get_inner_model(model)

    all_hs = []
    loader = DataLoader(TensorDataset(tokens), batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=4)
    model.eval()
    with torch.no_grad():
        for (batch_ids,) in tqdm(loader, desc="Hidden states (token input)"):
            batch_ids = batch_ids.to(device)
            mask = torch.ones_like(batch_ids)
            if inner is not None:
                out = inner(input_ids=batch_ids, attention_mask=mask,
                            output_hidden_states=False, return_dict=True,
                            use_cache=False)
                hs = out.last_hidden_state[:, -1, :].float()
            else:
                out = model(input_ids=batch_ids, attention_mask=mask,
                            output_hidden_states=True, return_dict=True,
                            use_cache=False)
                hs = out.hidden_states[-1][:, -1, :].float()
            all_hs.append(hs.cpu())
            del batch_ids, mask, out, hs

    return torch.cat(all_hs, dim=0)


def get_hidden_states_from_embeddings(model, emb_path, batch_size=32):
    """
    Forward pass random continuous embeddings through model → last hidden state.

    Bypasses the embedding layer: the same tensor produces comparable hidden
    states across models with the same hidden_size.

    Args:
        model: transformer model on GPU.
        emb_path: path to [N, seq_len, hidden_size] tensor.
        batch_size: inference batch size (auto-adjusted for large models).
    Returns:
        Tensor [N, hidden_size] (float32, CPU).
    """
    device = model.device
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
    hidden_size = getattr(model.config, "hidden_size", 0)
    batch_size = _adaptive_batch_size(hidden_size, batch_size, mode="embedding")
    inner = _get_inner_model(model)
    model_dtype = next(model.parameters()).dtype

    all_hs = []
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=4)
    model.eval()
    with torch.no_grad():
        for (batch_emb,) in tqdm(loader, desc="Hidden states (embedding input)"):
            batch_emb = batch_emb.to(device=device, dtype=model_dtype)
            mask = torch.ones(batch_emb.shape[:2], device=device)
            if inner is not None:
                out = inner(inputs_embeds=batch_emb, attention_mask=mask,
                            output_hidden_states=False, return_dict=True,
                            use_cache=False)
                hs = out.last_hidden_state[:, -1, :].float()
            else:
                out = model(inputs_embeds=batch_emb, attention_mask=mask,
                            output_hidden_states=True, return_dict=True,
                            use_cache=False)
                hs = out.hidden_states[-1][:, -1, :].float()
            all_hs.append(hs.cpu())

    return torch.cat(all_hs, dim=0)


def get_logits_from_embeddings(model, emb_path, batch_size=32):
    """
    Forward pass random embeddings through model → last-position logits.

    Args:
        model: transformer model on GPU.
        emb_path: path to [N, seq_len, hidden_size] tensor.
        batch_size: inference batch size.
    Returns:
        Tensor [N, vocab_size] (float32, CPU).
    """
    device = model.device
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
    hidden_size = getattr(model.config, "hidden_size", 0)
    batch_size = _adaptive_batch_size(hidden_size, batch_size, mode="embedding")
    model_dtype = next(model.parameters()).dtype

    all_logits = []
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=4)
    model.eval()
    with torch.no_grad():
        for (batch_emb,) in tqdm(loader, desc="Extracting logits"):
            batch_emb = batch_emb.to(device=device, dtype=model_dtype)
            mask = torch.ones(batch_emb.shape[:2], device=device)
            out = model(inputs_embeds=batch_emb, attention_mask=mask,
                        output_hidden_states=False, return_dict=True,
                        use_cache=False)
            all_logits.append(out.logits[:, -1, :].float().cpu())

    return torch.cat(all_logits, dim=0)


# ── Model Loading ────────────────────────────────────────────────────────────

def load_model(model_name_or_path, device_map="auto", trust_remote_code=True,
               hf_token=None):
    """
    Load a causal LM with automatic dtype promotion and attention backend.

    - float16 / bfloat16 models → bfloat16 + flash_attention_2
      (bfloat16's wider exponent range avoids NaN overflow with long random
       token sequences; FA2 provides efficient attention.)
    - float32 models → float32 + sdpa

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        device_map: "auto" (default) or explicit mapping.
        trust_remote_code: passed to from_pretrained.
        hf_token: HuggingFace auth token for gated models.
    Returns:
        model in eval mode on GPU.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    kwargs = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "device_map": device_map,
    }
    if hf_token:
        kwargs["token"] = hf_token

    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code,
            **({"token": hf_token} if hf_token else {})
        )
        native_dtype = getattr(config, "torch_dtype", None)
    except Exception:
        native_dtype = torch.float16

    if native_dtype in (torch.float16, torch.bfloat16):
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["attn_implementation"] = "flash_attention_2"
        logging.info(f"  bfloat16 + flash_attention_2 (native: {native_dtype})")
    else:
        kwargs["torch_dtype"] = torch.float32
        kwargs["attn_implementation"] = "sdpa"
        logging.info(f"  float32 + sdpa (native: {native_dtype})")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.eval()
    return model
