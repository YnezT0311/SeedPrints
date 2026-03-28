HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
CUDA_VISIBLE_DEVICES=2,3 python main.py \
    --benchmark_config 'config/benchmark_config.yaml' \
    --fingerprint_config 'config/reef.yaml' \
    --log_path 'logs/'