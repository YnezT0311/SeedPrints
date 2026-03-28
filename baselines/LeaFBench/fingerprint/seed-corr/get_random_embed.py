
import torch
import os
import numpy as np
from transformers import AutoModelForCausalLM

from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from transformers import (
    LlamaConfig, LlamaForCausalLM,
    MistralConfig, MistralForCausalLM,
    Qwen2Config, Qwen2ForCausalLM,
    # Add other model families as needed
)

def get_random_embed(model_name_or_path, hidden_size, save_path, num_sequences=10000, seq_length=1024):
        """
        Generate random embeddings to simulate random sequences of length seq_length.

        Args:
            model_name_or_path (str): The random generated embedding should match the model's embedding size.
            num_sequences (int): Number of random sequences to generate.
            seq_length (int): Length of each random sequence.
        Returns:

        """
        # Generate random embeddings
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
            del init_model

            # Generate random embeddings in batches to avoid CUDA memory issues
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