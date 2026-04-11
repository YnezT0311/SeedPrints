# SeedPrints

Official code for [**SeedPrints: Fingerprints Can Even Tell Which Seed Your Large Language Model Was Trained From**](https://arxiv.org/abs/2509.26404) (ICLR 2026).

SeedPrints is a model lineage detection method that determines whether a suspicious model B is derived from a given model A, not by relying on trained model family-wise behaviors but by leveraging the underlying bias originating from the base model — even the randomly initialized model. As a result, it significantly outperforms existing fingerprinting methods for detecting lineage across long pre-training stages. We consider this a biometric-like fingerprint: like a human fingerprint, it is something the model is born with at random initialization, and remains traceable across the entire lifecycle.

## How It Works

1. Feed shared random inputs (token sequences or random embeddings) through both models
2. Extract last-layer hidden states and identify the "identity dimensions" (i.e., the most biased output dimensions) of each model
3. Compute per-dimension Kendall tau correlation with softmax normalization between models' identity dimensions
4. Compare against an uncorrelated null distribution (we provide both empirical and analytical implementations) to determine whether the correlation is statistically significant

## Project Structure

```
SeedPrints/
├── seedprint.py               # Core algorithm
├── utils.py                   # Inference utilities
├── model_config.py            # Model registry (add your own models here)
├── test_toy_models.py         # Toy model experiments (Tables 1-4)
├── test_foundation_models.py  # Foundation model experiments (Table 5, Figure 3)
├── run_table1.sh              # Table 1: Different init seeds -> distinct fingerprints
├── run_table2.sh              # Table 2: Init fingerprint preserved after pre-training
├── run_table3.sh              # Table 3: Continual training does not confound fingerprint
├── run_table4.sh              # Table 4: Same data and data order, different seeds -> distinct fingerprints
├── run_table5.sh              # Table 5: Llama-2-7B fine-tune detection
├── run_figure3.sh             # Figure 3: OLMo-2-7B long pre-training detection
├── prepare_toy_models/        # Scripts to train toy models from scratch
└── baselines/                 # Baseline methods and LeaFBench integration
    ├── LeaFBench/             # Benchmark across 6 model families, 58 models
    ├── REEF-master/           # REEF baseline
    └── HuRef-main/            # HuRef baseline
```

## Quick Start

### Installation

```bash
# Option 1: Conda (recommended)
conda env create -f environment.yml
conda activate fingerprint

# Option 2: pip
pip install -r requirements.txt
```

### Example: Test if an OLMo-2-7B checkpoint shares lineage with an earlier checkpoint

```bash
CUDA_VISIBLE_DEVICES=0,1 python test_foundation_models.py \
    --target_model stage1-step928000-tokens3893B \
    --base_model stage1-step1000-tokens5B
```

## Reproducing Paper Results

### Toy Models (Tables 1–4)

We trained the small toy models (~160M parameters llama-style and qwen-style) with the training and finetuning scripts provided in `prepare_toy_models/`. You can choose to train your own (following `/prepare_toy_models/README.md`) or download our models from Huggingface. By default, toy models are automatically downloaded from [HuggingFace](https://huggingface.co/YnezT/SeedPrints-toy-models) if not found locally.

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
