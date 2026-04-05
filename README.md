# SeedPrints

Official code for **SeedPrints: Fingerprints Can Even Tell Which Seed Your Large Language Model Was Trained From** (ICLR 2026).

SeedPrint is a model lineage detection method that determines whether a suspicious model is derived from a given base model, using only random inputs and hidden state correlations — no training data, no fine-tuning, no watermark embedding required.

## How It Works

1. Feed shared random inputs (token IDs or continuous embeddings) through both models
2. Extract last-layer hidden states
3. Identify "identity dimensions" (bottom-k by mean activation) shared between the two models
4. Compute per-dimension Kendall tau correlation with softmax normalization
5. Z-score against an analytical null distribution — no simulation needed

If the p-value is below a significance threshold (e.g., 0.01), the two models share lineage.

## Project Structure

```
SeedPrints/
├── seedprint.py               # Core algorithm (identity extraction, correlation, z-score test)
├── utils.py                   # Inference utilities (random input generation, hidden state extraction)
├── model_config.py            # Model registry (HF model IDs, OLMo checkpoints)
├── test_toy_models.py         # Toy model experiments (Tables 1–4)
├── test_foundation_models.py  # Foundation model experiments (Table 5, Figure 3)
├── run_table1.sh              # Table 1: Different init seeds → distinct fingerprints
├── run_table2.sh              # Table 2: Init fingerprint preserved after pre-training
├── run_table3.sh              # Table 3: Continual training doesn't confound fingerprint
├── run_table4.sh              # Table 4: Same data, different seeds → distinct fingerprints
├── run_table5.sh              # Table 5: Llama-2-7B finetune detection
├── run_figure3.sh             # Figure 3: OLMo-2-7B training trajectory
└── baselines/                 # Baseline methods and LeaFBench integration
    ├── LeaFBench/             # Benchmark across 6 model families, 58 models
    ├── REEF-master/           # REEF baseline
    └── HuRef-main/            # HuRef baseline
```

## Quick Start

### Installation

```bash
pip install torch transformers scipy tqdm huggingface_hub
```

### Example: Test if llemma-7b is derived from Llama-2-7b

```bash
# Set your HuggingFace token for gated models (Llama family)
export HF_TOKEN="your_token_here"

CUDA_VISIBLE_DEVICES=0,1 python test_foundation_models.py \
    --target_model llemma-7b \
    --base_model Llama-2-7b
```

Expected output:
```
z_perdim: 153.7581
p_value: 0
k: 81
```

## Reproducing Paper Results

### Toy Models (Tables 1–4)

Toy models (160M parameters) are automatically downloaded from [HuggingFace](https://huggingface.co/YnezT/SeedPrints-toy-models) if not found locally.

```bash
bash run_table1.sh   # Different init seeds (expect p > 0.01)
bash run_table2.sh   # Init→Pretrained same seed (expect p < 0.01)
bash run_table3.sh   # Continual training (same lineage p<0.01, cross p>0.01)
bash run_table4.sh   # Cross-seed (expect p > 0.01)
```

Default setting: **embedding input + coset + per-dim only**.

### Foundation Models (Table 5)

```bash
bash run_table5.sh   # Llama-2-7B finetunes (expect p < 0.01)
```

Default setting: **token input + coset + per-dim only**.

### OLMo Training Trajectory (Figure 3)

```bash
bash run_figure3.sh  # OLMo-2-7B checkpoints vs final (expect p < 0.01)
```

### LeaFBench (58 models, 6 families)

See [baselines/LeaFBench/SeedPrints_README.md](baselines/LeaFBench/SeedPrints_README.md) for details.

```bash
cd baselines/LeaFBench
bash scripts/seed.sh
```

## Default Settings

| Experiment | Input | Identity Mode | Signal |
|------------|-------|--------------|--------|
| Toy models (Tables 1–4) | embedding | coset | per-dim |
| Foundation models (Table 5, Figure 3) | token | coset | per-dim |
| LeaFBench | token | coset | per-dim |

- **Token input**: Random token IDs go through each model's embedding layer, producing model-specific hidden states. Better at distinguishing architecturally similar families (e.g., Llama-3.1 vs Mistral).
- **Embedding input**: Random continuous embeddings bypass the embedding layer. Better for detecting init→pretrained lineage in toy models.
- **Coset**: Use the intersection of both models' bottom-k dimensions as identity dimensions.
- **Per-dim**: Per-column Kendall tau with softmax_T10 normalization → mean → z-score. Optional `--use_agg` adds a second signal with Bonferroni correction.

## Citation

```bibtex
@inproceedings{seedprints2026,
    title={SeedPrints: Fingerprints Can Even Tell Which Seed Your Large Language Model Was Trained From},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2026}
}
```
