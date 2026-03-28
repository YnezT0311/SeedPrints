import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader, TensorDataset

def _check_model_dtype_consistency(model):
    """
    Check if model has consistent dtypes across all parameters
    """
    dtypes = set()
    for param in model.parameters():
        dtypes.add(param.dtype)
    
    print(f"Model parameter dtypes found: {dtypes}")
    
    # Check if model has mixed dtypes (problematic)
    has_mixed_dtypes = len(dtypes) > 1
    
    # Get the most common dtype
    dtype_counts = {}
    for param in model.parameters():
        dtype_counts[param.dtype] = dtype_counts.get(param.dtype, 0) + 1
    
    primary_dtype = max(dtype_counts, key=dtype_counts.get)
    
    return {
        'has_mixed_dtypes': has_mixed_dtypes,
        'primary_dtype': primary_dtype,
        'all_dtypes': dtypes,
        'needs_conversion': has_mixed_dtypes or primary_dtype == torch.float16
    }

def _fix_model_dtype(model, target_dtype=torch.float32):
    """
    Convert entire model to target dtype to ensure consistency
    """
    print(f"Converting model to {target_dtype} for consistency")
    
    # Convert model parameters
    model = model.to(dtype=target_dtype)
    
    # Disable any caching that might cause issues
    if hasattr(model.config, 'use_cache'):
        original_cache = model.config.use_cache
        model.config.use_cache = False
        print("Disabled model caching to prevent dtype issues")
        return model, original_cache
    
    return model, None

def get_output(model, emb_path, output_type, batch_size, accelerator=None):
    if output_type == 'hidden_states':
        return _get_hidden_states_for_random_embeddings(model, emb_path, batch_size=batch_size, accelerator=accelerator)
    elif output_type == 'logits':
        return _get_logits_for_random_embeddings(model, emb_path, batch_size=batch_size, accelerator=accelerator)
    else:
        raise ValueError(f"Unknown output type: {output_type}")


def get_output_for_tokens(model, token_path, batch_size=32, accelerator=None):
    """
    Run model with random token IDs (input_ids) and return last-token hidden states.

    The token file contains integer IDs in [0, min_vocab_size) so they are valid
    for every model. Each model applies its own embedding layer first, making
    the hidden states model-specific.

    Args:
        model: loaded causal LM
        token_path: path to [N, seq_len] int64 tensor
        batch_size: forward-pass batch size
        accelerator: optional Accelerate accelerator

    Returns:
        Tensor of shape [N, hidden_size]  (float32)
    """
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device = model.device
    elif accelerator:
        device = accelerator.device
    else:
        device = model.device

    tokens = torch.load(token_path, map_location='cpu', mmap=True)
    print(f"Loaded random tokens from {token_path}  shape={tokens.shape}")

    hidden_size = model.config.hidden_size
    if hidden_size >= 4096:
        batch_size = min(batch_size, 8)

    dataset = TensorDataset(tokens)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=4)
    if accelerator:
        dataloader = accelerator.prepare(dataloader)

    local_hs = []
    model.eval()
    with torch.no_grad():
        for (batch_ids,) in tqdm.tqdm(dataloader, desc="Processing random tokens"):
            batch_ids = batch_ids.to(device)
            attention_mask = torch.ones_like(batch_ids)
            outputs = model(
                input_ids=batch_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            last_hs = outputs.hidden_states[-1][:, -1, :].float()
            local_hs.append(last_hs.cpu())
            del batch_ids, attention_mask, outputs, last_hs

    local_concat = torch.cat(local_hs, dim=0)
    if accelerator:
        local_concat = accelerator.gather(local_concat)
    print(f"Token hidden states shape: {local_concat.shape}")
    return local_concat


def _get_hidden_states_for_random_embeddings(model, emb_path, batch_size=32, accelerator=None):
    """
    Generate random embeddings and return all pre-projection head outputs

    Args:
        model: The transformer model
        emb_path: Path to embeddings file
        batch_size: Batch size for processing
        accelerator: Accelerate accelerator object

    Returns:
        all_hidden_states: Tensor of shape [num_samples, hidden_size]
    """
    hidden_size = model.config.hidden_size
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device = model.device
    elif accelerator:
        device = accelerator.device
    else:
        device = model.device
    random_embeddings = torch.load(emb_path, map_location='cpu', mmap=True)
    print(f"Loaded random embeddings from {emb_path} with shape {random_embeddings.shape}")
    
    # For larger models or large vocab models, use smaller batch sizes.
    # Large vocab (e.g. Gemma2 256k) makes logits expensive: bs*seq*vocab bytes.
    vocab_size = getattr(model.config, 'vocab_size', 0)
    if hidden_size >= 4096:
        batch_size = min(batch_size, 16)
    if vocab_size >= 100000:
        batch_size = min(batch_size, 4)
    
    all_hidden_states = []
    dataset = TensorDataset(random_embeddings)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True,
        num_workers=8
    )

    if accelerator:
        print(f"Original dataloader length: {len(dataloader)}")
        print(f"Process {accelerator.process_index}/{accelerator.num_processes}")
        dataloader = accelerator.prepare(dataloader)
        print(f"After prepare - GPU {accelerator.process_index} dataloader length: {len(dataloader)}")
        print(f"Expected samples per GPU: ~{len(random_embeddings) // accelerator.num_processes}")

    local_hidden_states = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Processing random embeddings"):
            model_dtype = next(model.parameters()).dtype
            batch_embeddings = batch[0].to(device=device, dtype=model_dtype)
            attention_mask = torch.ones(batch_embeddings.shape[:2], device=device, dtype=torch.bool)

            outputs = model(
                inputs_embeds=batch_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
            last_hidden_states = outputs.hidden_states[-1][:, -1, :].float()

            local_hidden_states.append(last_hidden_states.cpu())

            del batch_embeddings, attention_mask, outputs, last_hidden_states
    
    if local_hidden_states:
        local_concat = torch.cat(local_hidden_states, dim=0)

        if accelerator:
            all_hidden_states = accelerator.gather(local_concat)
        else:
            all_hidden_states = local_concat
        print(f"Final gathered hidden states shape: {all_hidden_states.shape}")

    return all_hidden_states

def _get_logits_for_random_embeddings(model, emb_path, batch_size=32, accelerator=None):
    """
    Generate random embeddings and return all pre-projection head outputs

    Args:
        model: The transformer model
        emb_path: Path to embeddings file
        batch_size: Batch size for processing
        accelerator: Accelerate accelerator object

    Returns:
        all_logits: Tensor of shape [num_samples, vocab_size]
    """
    hidden_size = model.config.hidden_size
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device = model.device
    elif accelerator:
        device = accelerator.device
    else:
        device = model.device
    random_embeddings = torch.load(emb_path, map_location='cpu')

    if hasattr(model.config, 'hidden_size') and model.config.hidden_size >= 4096:
        batch_size = min(batch_size, 8)
    
    all_logits = []
    dataset = TensorDataset(random_embeddings)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True,
        num_workers=16,
    )

    if accelerator:
        dataloader = accelerator.prepare(dataloader)

    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Processing random embeddings"):
            batch_embeddings = batch[0].to(device, non_blocking=True)
            attention_mask = torch.ones(batch_embeddings.shape[:2], device=device, dtype=torch.bool)
            
            outputs = model(
                inputs_embeds=batch_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
            logits = outputs.logits[:, -1, :]

            if accelerator:
                logits = accelerator.gather(logits)
            all_logits.append(logits.cpu())

    return torch.cat(all_logits, dim=0)

def get_output_for_co_indices(output_type, outputs, co_positions):
    if output_type == "hidden_states":
        return _get_hidden_states_for_co_positions(outputs, co_positions)
    elif output_type == "logits":
        return _get_logits_for_co_positions(outputs, co_positions)
    else:
        raise ValueError(f"Unknown output type: {output_type}")

def _get_hidden_states_for_co_positions(hidden_states, co_positions):
    """
    Get the hidden states for the co-positions.
    
    Args:
        hidden_states: List of tensors, each of shape [num_samples, hidden_size].
        co_positions: Set of token IDs for which to extract hidden states.
    
    Returns:
        co_hidden_states: List of tensors, each of shape [num_samples, len(co_positions)].
    """
    co_hidden_states = []
    for hidden_state in hidden_states:
        co_hidden_state = hidden_state[list(co_positions)]
        co_hidden_states.append(co_hidden_state)
    return co_hidden_states

def _get_logits_for_co_positions(logits, co_positions):
    """
    Get the logits for the co-positions.
    
    Args:
        logits: List of tensors, each of shape [num_samples, vocab_size].
        co_positions: Set of token IDs for which to extract logits.
    
    Returns:
        co_logits: List of tensors, each of shape [num_samples, len(co_positions)].
    """
    co_logits = []
    for logit in logits:
        co_logit = logit[list(co_positions)]
        co_logits.append(co_logit)
    return co_logits
