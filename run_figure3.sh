#!/bin/bash
# Figure 3: OLMo-2-7B checkpoint trajectory
# Each earlier checkpoint vs final checkpoint (expect p < 0.01)
# Default: token input + coset + perdim

final="stage1-step928000-tokens3893B"

checkpoints=(
    "stage1-step1000-tokens5B"
    "stage1-step207000-tokens869B"
    "stage1-step310000-tokens1301B"
    "stage1-step516000-tokens2165B"
    "stage1-step619000-tokens2597B"
    "stage1-step722000-tokens3029B"
    "stage1-step825000-tokens3461B"
)

# OLMo-7B needs ~2 GPUs in fp32; run 2 at a time
for ((i=0; i<${#checkpoints[@]}; i+=2)); do
    if [ $i -lt ${#checkpoints[@]} ]; then
        echo ">>> Figure 3: ${checkpoints[i]} vs ${final}"
        CUDA_VISIBLE_DEVICES=0,1,2,3 python test_foundation_models.py \
            --target_model "${final}" \
            --base_model "${checkpoints[i]}" \
            --num_samples 2000 --fingerprint_len 1024 &
    fi

    if [ $((i+1)) -lt ${#checkpoints[@]} ]; then
        echo ">>> Figure 3: ${checkpoints[i+1]} vs ${final}"
        CUDA_VISIBLE_DEVICES=4,5,6,7 python test_foundation_models.py \
            --target_model "${final}" \
            --base_model "${checkpoints[i+1]}" \
            --num_samples 2000 --fingerprint_len 1024 &
    fi

    wait
    echo "Round $((i/2 + 1)) complete."
done

echo "=== Figure 3 complete ==="
