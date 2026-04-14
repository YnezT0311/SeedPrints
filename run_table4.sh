#!/bin/bash
# Table 4: Continual training on diverse data doesn't confound fingerprint
# Same-lineage (seed 1000 matches): expect p < 0.01
# Cross-lineage (seed 123 ≠ 1000): expect p > 0.01
# Default: embedding input + coset + perdim

# Same-lineage
CUDA_VISIBLE_DEVICES=0,1 python test_toy_models.py \
    --target_model "TinyStoriesV2_cleaned-1000" \
    --base_model "openwebtext-1000" \
    --num_samples 10000 --fingerprint_len 1024 &

CUDA_VISIBLE_DEVICES=2,3 python test_toy_models.py \
    --target_model "code_stack-1000" \
    --base_model "openwebtext-1000" \
    --num_samples 10000 --fingerprint_len 1024 &

# Cross-lineage
CUDA_VISIBLE_DEVICES=4,5 python test_toy_models.py \
    --target_model "TinyStoriesV2_cleaned-123" \
    --base_model "openwebtext-1000" \
    --num_samples 10000 --fingerprint_len 1024 &

CUDA_VISIBLE_DEVICES=6,7 python test_toy_models.py \
    --target_model "code_stack-123" \
    --base_model "openwebtext-1000" \
    --num_samples 10000 --fingerprint_len 1024 &

wait
echo "=== Table 4 complete ==="
