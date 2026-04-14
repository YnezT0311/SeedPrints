#!/bin/bash
# Table 3: Same data+order, different seeds → distinct fingerprints (expect p > 0.01)
# Cross-seed: init-s_i as base, openwebtext-s_j as target (s_i ≠ s_j)
# Default: embedding input + coset + perdim

pairs=("1000:42" "42:123" "123:2000" "2000:1000")
gpu=0

for pair in "${pairs[@]}"; do
    IFS=':' read -r s_base s_target <<< "$pair"
    gpu_pair="${gpu},$((gpu+1))"
    echo ">>> Table 3: openwebtext-${s_target} vs init-${s_base} on GPU ${gpu_pair}"
    CUDA_VISIBLE_DEVICES=${gpu_pair} python test_toy_models.py \
        --target_model "openwebtext-${s_target}" \
        --base_model "init-${s_base}" \
        --num_samples 10000 --fingerprint_len 1024 &
    gpu=$(( (gpu + 2) % 8 ))
done
wait

echo "=== Table 3 complete ==="
