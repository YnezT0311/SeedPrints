#!/bin/bash
# Table 5: Llama-2-7B finetunes detected as same lineage (expect p < 0.01)
# Default: token input + coset + perdim

targets=(
    "meditron-7b"
    "llama-2-finance-7b"
    "vicuna-1.5-7b"
    "wizardmath-7b-v1.0"
    "codellama-7b"
    "llemma-7b"
)
base="Llama-2-7b"

for ((i=0; i<${#targets[@]}; i+=2)); do
    if [ $i -lt ${#targets[@]} ]; then
        echo ">>> Table 5: ${targets[i]} vs ${base} on GPU 0,1,2,3"
        CUDA_VISIBLE_DEVICES=0,1,2,3 python test_foundation_models.py \
            --target_model "${targets[i]}" \
            --base_model "${base}" \
            --num_samples 2000 --fingerprint_len 1024 &
    fi

    if [ $((i+1)) -lt ${#targets[@]} ]; then
        echo ">>> Table 5: ${targets[i+1]} vs ${base} on GPU 4,5,6,7"
        CUDA_VISIBLE_DEVICES=4,5,6,7 python test_foundation_models.py \
            --target_model "${targets[i+1]}" \
            --base_model "${base}" \
            --num_samples 2000 --fingerprint_len 1024 &
    fi

    wait
done

echo "=== Table 5 complete ==="
