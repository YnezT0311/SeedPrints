# SeedPrint Integration in LeaFBench

This document describes the SeedPrint-specific files, configuration, and how to reproduce results within the LeaFBench benchmark.

## File Structure

```
fingerprint/seed/
├── seed.py              # Main SeedPrint class (LLMFingerprintInterface)
├── correlation_test.py 
└── utils.py
```

### `seed.py`
The `SeedFingerprint` class implements the LeaFBench `LLMFingerprintInterface`, using **random token IDs** as input. Although SeedPrints supports both random tokens and random continuous embeddings, LeaFBench involves cross-family comparisons where models have different hidden sizes (e.g., Gemma-2b at 2304 vs Llama-7B at 4096). Using random embeddings would require different input tensors for different hidden sizes, introducing a clear family-wise bias (since the inputs themselves differ). To ensure a fair comparison, the LeaFBench integration uses only random token sequences (`[0, min_vocab)`), which are shared identically across all models regardless of architecture.

### `correlation_test.py`
Implements `test_lineage()` — the core SeedPrint hypothesis test:
1. Select bottom-k identity dimensions from mean activation profiles.
2. Apply softmax_T10 normalization to the selected dimensions.
3. Compute per-column Kendall tau correlation between two models.
4. Z-score against the analytical null distribution (no simulation needed).
5. Return a p-value (smaller = more evidence of shared lineage).

Supports `identity_mode="coset"` (intersection of both models' bottom-k) or `"base"` (use base model only).

Optional `use_agg=True` adds a second signal (per-sample mean → single Kendall tau) combined via max z-score with Bonferroni correction. This raises the p-value of borderline false positives while preserving power for strong true positives. Default is per-dim only.

### `utils.py`
- `get_random_tokens()`: Generate/cache random token ID sequences.
- `get_random_embed()`: Generate/cache random continuous embeddings.
- `get_output_for_tokens()`: Forward pass token IDs → last-token hidden states.
- `get_output()`: Forward pass embeddings → hidden states or logits.

## Configuration

**`config/seed.yaml`**:
```yaml
fingerprint_method: 'seed'
cached_fingerprints_path: 'cache/seed/seed_fingerprints.pth'
re_fingerprinting: False
fingerprint_type: 'white-box'
seed: 42

input_type: 'token'                         # "token" or "embedding"
random_input_path: 'cache/seed/fingerprint_input'
num_sequences: 2000                          # number of random sequences
seq_length: 1024                             # tokens per sequence
min_vocab_size: 32000                        # smallest vocab across all models (Llama-2)
batch_size: 4                                # inference batch size

ratio_k: 0.10                               # buffer_k = ratio_k × hidden_size
normalize: 'softmax_T10'                     # row-wise softmax with temperature 10
identity_mode: 'coset'                       # "coset" or "base"
use_agg: False                               # include agg signal (default: per-dim only)
```

## How to Run

### Quick start

We provide pre-computed hidden states and random token sequences for fast reproduction. Download them from HuggingFace:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('YnezT0311/SeedPrints-cache', 'seed_fingerprints.pth',
                local_dir='cache/seed/')
hf_hub_download('YnezT0311/SeedPrints-cache', 'random_tokens_2000_1024_vocab32000.pt',
                local_dir='cache/seed/fingerprint_input/')
"
```

Then run the benchmark (auto-detects free GPUs):
```bash
bash scripts/seed.sh
```

### Re-run inference (reuse random inputs)

To regenerate hidden states from scratch while reusing the same random token sequences:

```bash
# Remove cached fingerprints; random tokens will be reused automatically
rm -f cache/seed/seed_fingerprints.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --benchmark_config 'config/benchmark_config.yaml' \
    --fingerprint_config 'config/seed.yaml' \
    --log_path 'logs/seed/'
```

### Fully from scratch

To regenerate everything including random token sequences:

```bash
# Remove all cached artifacts
rm -rf cache/seed/

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --benchmark_config 'config/benchmark_config.yaml' \
    --fingerprint_config 'config/seed.yaml' \
    --log_path 'logs/seed/'
```

## Expected Runtime

| Setting | GPUs | Time |
|---------|------|------|
| From scratch (58 models, 2000 samples each) | 4× A40 (48GB) | ~6 hours |
| With cached fingerprints | 1× GPU | ~5 minutes |

The bottleneck is model loading + forward pass. Each 7B model takes ~4–5 minutes (load + 2000 forward passes at batch_size=4). The comparison/evaluation step takes only seconds.

## Expected Output

### Overall Metrics (Default Setup)
```
AUC: 0.9966, Accuracy: 0.9943
TPR: 1.0000, TNR: 0.9931, FPR: 0.0069, FNR: 0.0000
TPR@1%FPR: 1.0000
```

### Per-Family AUC
| Model Family | AUC (pretrained) | AUC (instruct) |
|--------------|-----------------|----------------|
| Qwen-2.5-7B | 0.989 | 0.989 |
| Qwen2.5-14B | 0.990 | 0.990 |
| Llama-3.1-8B | 1.000 | 1.000 |
| Mistral-7B-v0.3 | 1.000 | 1.000 |
| Gemma-2-2b | 1.000 | 1.000 |
| Llama-2-7b | 1.000 | 1.000 |

### Cached Artifacts
After a full run, the following files are saved:
- `cache/seed/seed_fingerprints.pth` (~1.7 GB): Hidden state tensors for all 58 models.
- `cache/seed/fingerprint_input/random_tokens_2000_1024_vocab32000.pt` (~8 MB): Shared random token sequences.

These can be reused for fast reproduction without re-running inference.
