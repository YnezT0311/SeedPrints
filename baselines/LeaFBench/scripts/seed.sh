# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
# CUDA_VISIBLE_DEVICES=2,3 python main.py \
#     --benchmark_config 'config/benchmark_config.yaml' \
#     --fingerprint_config 'config/seed.yaml' \
#     --log_path 'logs/'


# Configuration
MIN_FREE_GPUS=3
MEMORY_THRESHOLD=6000  # MB - consider GPU free if memory usage < this
CHECK_INTERVAL=180     # seconds between checks

echo "Monitoring GPU usage - waiting for $MIN_FREE_GPUS free GPUs..."
echo "Memory threshold: ${MEMORY_THRESHOLD}MB"
echo "Check interval: ${CHECK_INTERVAL}s"

while true; do
    # Get GPU memory usage in MB
    gpu_memory=($(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits))
    
    free_gpu_count=0
    free_gpus=()
    
    for i in "${!gpu_memory[@]}"; do
        if [ "${gpu_memory[$i]}" -lt "$MEMORY_THRESHOLD" ]; then
            free_gpu_count=$((free_gpu_count + 1))
            free_gpus+=($i)
        fi
    done
    
    echo "$(date): Free GPUs: $free_gpu_count (IDs: ${free_gpus[*]})"
    
    if [ "$free_gpu_count" -ge "$MIN_FREE_GPUS" ]; then
        echo "Found $free_gpu_count free GPUs! Starting job..."
        
        # Set visible GPUs to all 4 free ones
        export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${free_gpus[*]:0:$MIN_FREE_GPUS}")
        echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
        
        # Run your original command
        HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python main.py \
            --benchmark_config 'config/benchmark_config.yaml' \
            --fingerprint_config 'config/seed.yaml' \
            --log_path 'logs/'
        
        echo "Job completed!"
        exit 0
    else
        echo "Only $free_gpu_count GPUs available, need $MIN_FREE_GPUS. Waiting..."
        sleep "$CHECK_INTERVAL"
    fi
done