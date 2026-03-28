import torch
import os


def get_random_tokens(save_path, num_sequences=2000, seq_length=1024, min_vocab_size=32000):
    """
    Generate random token ID sequences shared across all models.

    Uses the minimum vocabulary size so every model can embed the tokens.

    Args:
        save_path: directory to save the token file.
        num_sequences: number of random sequences.
        seq_length: length of each sequence.
        min_vocab_size: upper bound for token IDs (exclusive).

    Returns:
        path to saved token file  [num_sequences, seq_length]  int64.
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
