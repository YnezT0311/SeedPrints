#!/bin/bash
# Table 1: Different init seeds → distinct fingerprints (expect p > 0.01)
# Default: embedding input + coset + perdim

pairs=("42:2000" "123:42" "1000:123" "2000:1000")
gpu=0

for pair in "${pairs[@]}"; do
    IFS=':' read -r s_i s_j <<< "$pair"
    echo ">>> Table 1: init-${s_i} vs init-${s_j} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} python test_toy_models.py \
        --target_model "init-${s_j}" \
        --base_model "init-${s_i}" \
        --num_samples 10000 --fingerprint_len 1024 &
    gpu=$(( (gpu + 1) % 8 ))
done
wait

echo "=== Table 1 complete ==="
