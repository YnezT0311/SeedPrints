"""
utils.py - Inference utilities for SeedPrint within LeaFBench.

Provides:
  - Random token sequence generation
  - Hidden state extraction from transformer models
"""

import os
import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset


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


# ── Hidden State Extraction ──────────────────────────────────────────────────

def get_output(model, token_path, batch_size=32, accelerator=None):
    """
    Forward pass random token IDs through model, return last-token hidden states.

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
        for (batch_ids,) in tqdm.tqdm(loader, desc="Extracting hidden states"):
            batch_ids = batch_ids.to(device)
            mask = torch.ones_like(batch_ids)
            out = model(input_ids=batch_ids, attention_mask=mask,
                        output_hidden_states=True, return_dict=True, use_cache=False)
            all_hs.append(out.hidden_states[-1][:, -1, :].float().cpu())
            del batch_ids, mask, out

    result = torch.cat(all_hs, dim=0)
    if accelerator:
        result = accelerator.gather(result)
    print(f"Hidden states shape: {result.shape}")
    return result


def _get_device(model, accelerator):
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        return model.device
    if accelerator:
        return accelerator.device
    return model.device
