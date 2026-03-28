#!/bin/bash
# Table 2: Init fingerprint preserved after pre-training (expect p < 0.01)
# Default: embedding input + coset + perdim

seeds=(42 123 1000 2000)
gpu=0

for s in "${seeds[@]}"; do
    gpu_pair="${gpu},$((gpu+1))"
    echo ">>> Table 2: openwebtext-${s} vs init-${s} on GPU ${gpu_pair}"
    CUDA_VISIBLE_DEVICES=${gpu_pair} python test_toy_models.py \
        --target_model "openwebtext-${s}" \
        --base_model "init-${s}" \
        --num_samples 10000 --fingerprint_len 1024 &
    gpu=$(( (gpu + 2) % 8 ))
done
wait

echo "=== Table 2 complete ==="
