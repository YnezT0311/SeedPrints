import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm


def get_random_embed(model_name_or_path, hidden_size, save_path, num_sequences=10000, seq_length=1024):
    """
    Generate random embeddings ~ N(mean, std) from a randomly-initialized model
    with the same architecture as model_name_or_path.

    Embeddings are shared across all models with the same hidden_size.

    Args:
        model_name_or_path: HF model path (used to get architecture config).
        hidden_size: model hidden size.
        save_path: directory to save the embedding file.
        num_sequences: number of random sequences to generate.
        seq_length: length of each random sequence.

    Returns:
        path to the saved embedding file.
    """
    save_path = os.path.join(save_path, f'random_embed_{num_sequences}_{seq_length}_{hidden_size}.pt')
    if os.path.exists(save_path):
        print(f"Existing random embeddings at {save_path}")
    else:
        config = AutoConfig.from_pretrained(model_name_or_path)
        init_model = AutoModelForCausalLM.from_config(config)
        print(f"Generating new random embeddings and saving to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        embedding_std = init_model.model.embed_tokens.weight.std().item()
        embedding_mean = init_model.model.embed_tokens.weight.mean().item()
        print(f"Embedding stats: mean={embedding_mean:.6f}, std={embedding_std:.6f}")
        del init_model

        # Use a fixed seed based on hidden_size for reproducibility
        torch.manual_seed(42 + hidden_size)
        batch_size = 1000
        random_embeddings = []

        for i in tqdm(range(0, num_sequences, batch_size), desc="Generating random embeddings"):
            current_batch_size = min(batch_size, num_sequences - i)
            batch_embeddings = torch.normal(
                mean=embedding_mean,
                std=embedding_std,
                size=(current_batch_size, seq_length, hidden_size),
            )
            random_embeddings.append(batch_embeddings)

        random_embeddings = torch.cat(random_embeddings, dim=0)
        torch.save(random_embeddings, save_path)
        print(f"Random embeddings saved to {save_path}")

    return save_path
