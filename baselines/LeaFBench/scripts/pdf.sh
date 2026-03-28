HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
CUDA_VISIBLE_DEVICES=6,7 python main.py \
    --benchmark_config 'config/benchmark_config.yaml' \
    --fingerprint_config 'config/pdf.yaml' \
    --log_path 'logs/'