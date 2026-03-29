"""
utils.py - Inference utilities for SeedPrint within LeaFBench.

Provides:
  - Random input generation (token sequences and continuous embeddings)
  - Hidden state extraction from transformer models
"""

import os
import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoConfig


# ── Random Input Generation ──────────────────────────────────────────────────

def get_random_tokens(save_path, num_sequences=2000, seq_length=1024, min_vocab_size=32000):
    """
    Generate or load cached random token ID sequences.

    Token IDs are in [0, min_vocab_size) so every model can embed them.

    Returns:
        path to [num_sequences, seq_length] int64 tensor on disk.
    """
    fname = os.path.join(save_path, f'random_tokens_{num_sequences}_{seq_length}_vocab{min_vocab_size}.pt')
    if os.path.exists(fname):
        print(f"Existing random tokens at {fname}")
    else:
        os.makedirs(save_path, exist_ok=True)
        torch.manual_seed(42)
        tokens = torch.randint(0, min_vocab_size, (num_sequences, seq_length), dtype=torch.long)
        torch.save(tokens, fname)
        print(f"Random tokens saved to {fname}  shape={tokens.shape}")
    return fname


def get_random_embed(model_name_or_path, hidden_size, save_path,
                     num_sequences=10000, seq_length=1024):
    """
    Generate or load cached random embeddings ~ N(μ, σ) from a randomly-
    initialized model with the given architecture.

    Embeddings are shared across all models with the same hidden_size.

    Returns:
        path to [num_sequences, seq_length, hidden_size] tensor on disk.
    """
    fpath = os.path.join(save_path, f'random_embed_{num_sequences}_{seq_length}_{hidden_size}.pt')
    if os.path.exists(fpath):
        print(f"Existing random embeddings at {fpath}")
    else:
        config = AutoConfig.from_pretrained(model_name_or_path)
        init_model = AutoModelForCausalLM.from_config(config)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        mu = init_model.model.embed_tokens.weight.mean().item()
        sigma = init_model.model.embed_tokens.weight.std().item()
        print(f"Embedding stats: mean={mu:.6f}, std={sigma:.6f}")
        del init_model

        torch.manual_seed(42 + hidden_size)
        batch_size = 1000
        parts = []
        for i in tqdm.tqdm(range(0, num_sequences, batch_size), desc="Generating embeddings"):
            bs = min(batch_size, num_sequences - i)
            parts.append(torch.normal(mu, sigma, size=(bs, seq_length, hidden_size)))
        torch.save(torch.cat(parts, dim=0), fpath)
        print(f"Random embeddings saved to {fpath}")
    return fpath


# ── Hidden State Extraction ──────────────────────────────────────────────────

def get_output_for_tokens(model, token_path, batch_size=32, accelerator=None):
    """
    Forward pass random token IDs through model → last-token hidden states.

    Each model applies its own embedding layer, producing model-specific
    hidden states.

    Returns:
        Tensor [N, hidden_size] (float32).
    """
    device = _get_device(model, accelerator)
    tokens = torch.load(token_path, map_location='cpu', mmap=True)
    print(f"Loaded random tokens from {token_path}  shape={tokens.shape}")

    hidden_size = model.config.hidden_size
    if hidden_size >= 4096:
        batch_size = min(batch_size, 8)

    loader = DataLoader(TensorDataset(tokens), batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=4)
    if accelerator:
        loader = accelerator.prepare(loader)

    all_hs = []
    model.eval()
    with torch.no_grad():
        for (batch_ids,) in tqdm.tqdm(loader, desc="Hidden states (token input)"):
            batch_ids = batch_ids.to(device)
            mask = torch.ones_like(batch_ids)
            out = model(input_ids=batch_ids, attention_mask=mask,
                        output_hidden_states=True, return_dict=True, use_cache=False)
            all_hs.append(out.hidden_states[-1][:, -1, :].float().cpu())
            del batch_ids, mask, out

    result = torch.cat(all_hs, dim=0)
    if accelerator:
        result = accelerator.gather(result)
    print(f"Token hidden states shape: {result.shape}")
    return result


def get_output(model, emb_path, output_type='hidden_states', batch_size=32,
               accelerator=None):
    """
    Forward pass random embeddings through model → hidden states or logits.

    Returns:
        Tensor [N, hidden_size] or [N, vocab_size] (float32).
    """
    device = _get_device(model, accelerator)
    embeddings = torch.load(emb_path, map_location='cpu', mmap=True)
    print(f"Loaded random embeddings from {emb_path}  shape={embeddings.shape}")

    hidden_size = model.config.hidden_size
    vocab_size = getattr(model.config, 'vocab_size', 0)
    if hidden_size >= 4096:
        batch_size = min(batch_size, 16)
    if vocab_size >= 100000:
        batch_size = min(batch_size, 4)

    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=8)
    if accelerator:
        loader = accelerator.prepare(loader)

    model_dtype = next(model.parameters()).dtype
    all_out = []
    model.eval()
    with torch.no_grad():
        for (batch_emb,) in tqdm.tqdm(loader, desc=f"Extracting {output_type}"):
            batch_emb = batch_emb.to(device=device, dtype=model_dtype)
            mask = torch.ones(batch_emb.shape[:2], device=device, dtype=torch.bool)
            out = model(inputs_embeds=batch_emb, attention_mask=mask,
                        output_hidden_states=True, return_dict=True, use_cache=False)
            if output_type == 'hidden_states':
                all_out.append(out.hidden_states[-1][:, -1, :].float().cpu())
            else:
                all_out.append(out.logits[:, -1, :].float().cpu())
            del batch_emb, mask, out

    result = torch.cat(all_out, dim=0)
    if accelerator:
        result = accelerator.gather(result)
    print(f"Output shape: {result.shape}")
    return result


def _get_device(model, accelerator):
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        return model.device
    if accelerator:
        return accelerator.device
    return model.device
