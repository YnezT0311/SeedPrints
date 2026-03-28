import torch
import numpy as np
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


# ------------ helper functions --------

def _get_layers(model):
    # Works for LlamaForCausalLM, Qwen2ForCausalLM, etc.
    mm = getattr(model, "model", None) or getattr(model, "transformer", None)
    if mm is None or not hasattr(mm, "layers"):
        raise ValueError("Could not find model.model.layers (or .transformer.layers).")
    return mm.layers

def _get_attn_module(layer):
    # Llama/Qwen usually use `self_attn`
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
    if attn is None:
        raise ValueError("Layer has no attention module (self_attn/attn).")
    return attn

def _get_proj(attn, name):
    # Extract q_proj/k_proj/v_proj/o_proj
    if hasattr(attn, name):
        mod = getattr(attn, name)
        # prefer .weight if it's a Linear
        if hasattr(mod, "weight"):
            return mod.weight
        # some impls keep packed params differently
        return mod
    # Fallback: search by exact name in named_parameters
    for n, p in attn.named_parameters(recurse=False):
        if n == f"{name}.weight":
            return p
    raise AttributeError(f"Attention missing projection: {name}")

def get_std_sequences(model, use_fp32=True):
    """
    Returns dict with four np arrays (length = #layers), each is normalized:
      {'Q': zQ, 'K': zK, 'V': zV, 'O': zO, 'raw': {'Q': rawQ, ...}}
    """
    model.eval()
    layers = _get_layers(model)

    q_std, k_std, v_std, o_std = [], [], [], []
    with torch.no_grad():
        for layer in layers:
            attn = _get_attn_module(layer)
            Q = _get_proj(attn, "q_proj")
            K = _get_proj(attn, "k_proj")
            V = _get_proj(attn, "v_proj")
            O = _get_proj(attn, "o_proj")

            def tstd(w):
                t = w
                if use_fp32:
                    t = t.float()
                # std over all elements
                return torch.std(t.detach()).item()

            q_std.append(tstd(Q))
            k_std.append(tstd(K))
            v_std.append(tstd(V))
            o_std.append(tstd(O))

    raw = {
        'Q': np.array(q_std, dtype=np.float64),
        'K': np.array(k_std, dtype=np.float64),
        'V': np.array(v_std, dtype=np.float64),
        'O': np.array(o_std, dtype=np.float64),
    }

    def zscore(x):
        mu, sd = x.mean(), x.std()
        # guard against zero-variance sequence
        return (x - mu) / (sd if sd > 0 else 1.0)

    z = {k: zscore(v) for k, v in raw.items()}
    z['raw'] = raw
    return z

def corr(a, b):
    # Pearson correlation for 1D arrays
    if a.size != b.size:
        raise ValueError("Use corr_interp() when lengths differ.")
    if a.size < 2:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def corr_interp(a, b):
    """
    Correlate two sequences of possibly different lengths via linear interpolation
    (interpolate the shorter onto the longer's index).
    """
    La, Lb = len(a), len(b)
    if La == Lb:
        return corr(a, b)
    if La < Lb:
        x = np.linspace(0, La - 1, num=La)
        x_new = np.linspace(0, La - 1, num=Lb)
        a_i = np.interp(x_new, x, a)
        return corr(a_i, b)
    else:
        x = np.linspace(0, Lb - 1, num=Lb)
        x_new = np.linspace(0, Lb - 1, num=La)
        b_i = np.interp(x_new, x, b)
        return corr(a, b_i)

def prepare_model_and_tokenizer(tokenizer, model_type: str, model_name: str, init_seed: int, abs_dir="."):
    if model_name == "init":
        torch.manual_seed(init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(init_seed)

        if os.path.exists(os.path.join(abs_dir, '..', f"model/init-{model_type}-seed-{init_seed}")):
            model_path = os.path.join(abs_dir, '..', f"model/init-{model_type}-seed-{init_seed}")
            if model_type == "qwen":
                from transformers import Qwen2ForCausalLM
                model = Qwen2ForCausalLM.from_pretrained(model_path)
            elif model_type == "llama":
                from transformers import LlamaForCausalLM
                model = LlamaForCausalLM.from_pretrained(model_path)
        else:
            if model_type == "qwen":
                from transformers import Qwen2Config, Qwen2ForCausalLM
                config = Qwen2Config(
                    hidden_size=768,
                    intermediate_size=3072,
                    num_attention_heads=12,
                    num_hidden_layers=12,
                    vocab_size=tokenizer.vocab_size,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_position_embeddings=2048,
                    rope_theta=10000.0,                    # Qwen2 default
                    tie_word_embeddings=False,             # Qwen2 default: untied embeddings
                    use_cache=False
                )
                model = Qwen2ForCausalLM(config)
                model.save_pretrained(os.path.join(abs_dir, '..', f"model/init-qwen-seed-{init_seed}"))
                tokenizer.save_pretrained(os.path.join(abs_dir, '..', f"model/init-qwen-seed-{init_seed}"))
            else:
                from transformers import LlamaConfig, LlamaForCausalLM
                config = LlamaConfig(
                    hidden_size=768,
                    intermediate_size=2048,
                    num_attention_heads=12,
                    num_hidden_layers=12,
                    vocab_size=tokenizer.vocab_size,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_position_embeddings=2048,
                )
                model = LlamaForCausalLM(config)
                model.save_pretrained(os.path.join(abs_dir, '..', f"model/init-llama-seed-{init_seed}"))
                tokenizer.save_pretrained(os.path.join(abs_dir, '..', f"model/init-llama-seed-{init_seed}"))
    else:
        model_path = f"model/{model_name}-seed-{init_seed}"
        if not os.path.exists(os.path.join(abs_dir, '..', model_path)):
            # load checkpoints, e.g., "llama-160M-finetune-BabyLM/checkpoint-500"
            model_names = model_name.split('/')
            model_path = f"{model_names[0]}-seed-{init_seed}/{model_names[1]}"
        if model_type == "qwen":
            from transformers import Qwen2ForCausalLM
            model = Qwen2ForCausalLM.from_pretrained(os.path.join(abs_dir, '..', model_path))
        else:
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(os.path.join(abs_dir, '..', model_path))
    
    model.eval().cuda()
    return model

# -------- public APIs --------

def between_models_correlation(zA, zB):
    """
    Type-wise (Q with Q, K with K, ...) Pearson correlation after interpolation if needed.
    Returns dict: {'Q': rQ, 'K': rK, 'V': rV, 'O': rO, 'overall_mean': avg}
    """
    out = {}
    for key in ['Q', 'K', 'V', 'O']:
        out[key] = corr_interp(np.asarray(zA[key]), np.asarray(zB[key]))
    out['overall_mean'] = float(np.mean([out[k] for k in ['Q','K','V','O']]))
    return out

def main(target_model_name, ref_model_name, args):
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    abs_dir = os.path.abspath(os.path.join(abs_dir, '..'))
    if args.model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
    if "/" in target_model_name:
        suspected_model = AutoModelForCausalLM.from_pretrained(target_model_name,trust_remote_code=True).eval()
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name,trust_remote_code=True).eval()
    else:
        suspected_model = prepare_model_and_tokenizer(tokenizer, model_type=args.model_type, model_name=target_model_name, init_seed=args.target_model_seed, abs_dir=abs_dir)
        ref_model = prepare_model_and_tokenizer(tokenizer, model_type=args.model_type, model_name=ref_model_name, init_seed=args.ref_model_seed, abs_dir=abs_dir)

    z_suspected = get_std_sequences(suspected_model, use_fp32=True)
    z_ref = get_std_sequences(ref_model, use_fp32=True)

    cross = between_models_correlation(z_suspected, z_ref)
    print("\nBetween-model correlation:")
    print({k: round(v, 3) for k, v in cross.items()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate intrinsic fingerprint correlation between two models.")
    # parser.add_argument("--target_model_name", type=str, default='init-llama-160M', help="Name of the suspected target model.")
    # parser.add_argument("--ref_model_name", type=str, default='llama-160M-fullset-openwebtext', help="Name of the reference protected model.")
    parser.add_argument("--target_model_name", type=str, default='EleutherAI/llemma_7b', help="Name of the suspected target model.")
    parser.add_argument("--ref_model_name", type=str, default='meta-llama/Llama-2-7b-hf', help="Name of the reference protected model.")
    parser.add_argument("--target_model_seed", type=int, default=2000, help="Seed for the target model.")
    parser.add_argument("--ref_model_seed", type=int, default=2000, help="Seed for the reference model.")
    parser.add_argument("--model_type", type=str, default="llama", choices=["llama", "qwen"], help="Type of the models (llama or qwen).")
    args = parser.parse_args()
    
    main(args.target_model_name, args.ref_model_name, args)

    
