"""
model_config.py - Model registry for SeedPrint experiments.

Maps short tags to HuggingFace model IDs. Add new models here.
"""

import os


# ── HuggingFace auth token (for gated models like Llama) ─────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", None)


# ── Foundation models: short tag → HuggingFace model ID ──────────────────────

FOUNDATION_MODELS = {
    # Llama-2 base
    "Llama-2-7b":                   "meta-llama/Llama-2-7b-hf",
    "Llama-2-7b-chat":              "meta-llama/Llama-2-7b-chat-hf",
    # Llama-2 fine-tunes (Table 5)
    "meditron-7b":                  "epfl-llm/meditron-7b",
    "llama-2-finance-7b":           "cxllin/Llama2-7b-Finance",
    "vicuna-1.5-7b":                "lmsys/vicuna-7b-v1.5",
    "wizardmath-7b-v1.0":           "WizardLMTeam/WizardMath-7B-V1.0",
    "codellama-7b":                 "codellama/CodeLlama-7b-hf",
    "llemma-7b":                    "EleutherAI/llemma_7b",
    # Llama-3.1
    "Llama-3.1-8B":                 "meta-llama/Llama-3.1-8B",
    "Llama-3.1-8B-Instruct":        "meta-llama/Llama-3.1-8B-Instruct",
    # Qwen
    "Qwen-2.5-7B":                  "Qwen/Qwen2.5-7B",
    "Qwen-2.5-14B":                 "Qwen/Qwen2.5-14B",
    # Gemma
    "Gemma-2-2b":                   "google/gemma-2-2b",
    "Gemma-2-2b-it":                "google/gemma-2-2b-it",
    # Mistral
    "Mistral-7B-v0.3":              "mistralai/Mistral-7B-v0.3",
}


# ── OLMo-2-7B (Figure 3) ────────────────────────────────────────────────────

OLMO_BASE_MODEL = "allenai/OLMo-2-1124-7B"

OLMO_CHECKPOINTS = [
    "stage1-step1000-tokens5B",
    "stage1-step207000-tokens869B",
    "stage1-step310000-tokens1301B",
    "stage1-step413000-tokens1733B",
    "stage1-step516000-tokens2165B",
    "stage1-step619000-tokens2597B",
    "stage1-step722000-tokens3029B",
    "stage1-step825000-tokens3461B",
    "stage1-step928000-tokens3893B",
]


# ── Embedding-init factories (by hidden_size) ────────────────────────────────
# Only the embedding layer statistics (mean, std) are used; the rest is
# discarded immediately. Required only for --input_type embedding.

def _make_init_fn():
    from transformers import (
        LlamaConfig, LlamaForCausalLM,
        GemmaConfig, GemmaForCausalLM,
        Qwen2Config, Qwen2ForCausalLM,
    )
    return {
        2048: lambda: LlamaForCausalLM(LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")),
        2304: lambda: GemmaForCausalLM(GemmaConfig.from_pretrained("google/gemma-2-2b")),
        3584: lambda: Qwen2ForCausalLM(Qwen2Config.from_pretrained("Qwen/Qwen2.5-7B")),
        4096: lambda: LlamaForCausalLM(LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")),
        5120: lambda: Qwen2ForCausalLM(Qwen2Config.from_pretrained("Qwen/Qwen2.5-14B-Instruct")),
    }

DIM_INIT_FN = _make_init_fn()
