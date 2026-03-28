"""
Test cross-family correlation under different conditions:
1. N(0,1) embeddings vs current N(0,0.02)
2. Coset mode vs base mode

Results saved to results_crossfamily.txt
"""
import torch
import numpy as np
from scipy.stats import kendalltau
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
import time, gc

RESULTS_FILE = "results_crossfamily.txt"

def log(msg):
    print(msg, flush=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(msg + "\n")

def get_bot_idx(hs, bk):
    avg = hs.mean(dim=0)
    return sorted(torch.topk(avg, bk, largest=False).indices.tolist())

def perdim_corr(a, b, idx, T=10, max_dims=50):
    an = F.softmax(a[:, idx] / T, dim=1)
    bn = F.softmax(b[:, idx] / T, dim=1)
    taus = []
    for i in range(min(max_dims, len(idx))):
        t, _ = kendalltau(an[:, i].numpy(), bn[:, i].numpy())
        if not np.isnan(t):
            taus.append(t)
    return np.mean(taus) if taus else 0.0

def agg_corr(a, b, idx):
    aa = a[:, idx].mean(dim=1).numpy()
    bb = b[:, idx].mean(dim=1).numpy()
    t, _ = kendalltau(aa, bb)
    return t

def run_model(path, embeddings):
    """Load model, run forward pass, return HS [N, D]."""
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    native_dtype = getattr(cfg, 'torch_dtype', torch.bfloat16)
    is_gemma = "gemma" in path.lower()

    load_kwargs = {
        "device_map": "auto" if is_gemma else "balanced",
        "trust_remote_code": True,
    }
    if native_dtype in (torch.float16, torch.bfloat16) and not is_gemma:
        load_kwargs["torch_dtype"] = native_dtype
        load_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        load_kwargs["torch_dtype"] = native_dtype if native_dtype in (torch.float16, torch.bfloat16) else torch.float32
        load_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    model.eval()

    model_dtype = next(model.parameters()).dtype
    device = model.device
    log(f"  Loaded with dtype={model_dtype}, device={device}")

    all_hs = []
    N = embeddings.shape[0]
    bs = 16
    with torch.no_grad():
        for i in range(0, N, bs):
            batch = embeddings[i:i+bs].to(device=device, dtype=model_dtype)
            mask = torch.ones(batch.shape[:2], device=device, dtype=torch.bool)
            out = model(inputs_embeds=batch, attention_mask=mask,
                        output_hidden_states=True, return_dict=True, use_cache=False)
            hs = out.hidden_states[-1][:, -1, :].float().cpu()
            all_hs.append(hs)
            del batch, mask, out, hs
            if i % 320 == 0:
                log(f"    {i}/{N}")

    result = torch.cat(all_hs, dim=0)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def compare_pairs(hs_dict, pairs, bk, mode="base"):
    """Compare pairs with base or coset mode."""
    for base_name, target_name in pairs:
        a = hs_dict[base_name]
        b = hs_dict[target_name]
        idx_a = get_bot_idx(a, bk)

        if mode == "coset":
            idx_b = get_bot_idx(b, bk)
            idx = sorted(list(set(idx_a) & set(idx_b)))
            k = len(idx)
        else:
            idx = idx_a
            k = len(idx)

        if k == 0:
            log(f"  {base_name:20s} -> {target_name:20s}  k=0 (no overlap)")
            continue

        pd = perdim_corr(a, b, idx)
        ag = agg_corr(a, b, idx)
        log(f"  {base_name:20s} -> {target_name:20s}  perdim={pd:.4f}  agg={ag:.4f}  k={k}")


def main():
    # Clear results file
    with open(RESULTS_FILE, "w") as f:
        f.write(f"Cross-family correlation test - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    models_info = {
        "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
        "Mistral-7B-v0.3": "mistralai/Mistral-7B-v0.3",
        "Llama-2-7b": "meta-llama/Llama-2-7b-hf",
    }

    bk = 410
    cross_pairs = [
        ("Llama-3.1-8B", "Mistral-7B-v0.3"),
        ("Llama-3.1-8B", "Llama-2-7b"),
        ("Mistral-7B-v0.3", "Llama-2-7b"),
    ]
    # Also add reverse pairs for coset (coset is symmetric but base mode isn't)
    all_pairs = cross_pairs + [(b, a) for a, b in cross_pairs]

    # ============================================================
    # Part 0: Current results with N(0, 0.02) cached HS
    # ============================================================
    log("Part 0: Current N(0, 0.02) embeddings - BASE mode")
    log("-" * 60)
    data = torch.load('cache/seed/seed_fingerprints.pth', map_location='cpu')
    hs_002 = {name: data[name].float() for name in models_info}
    compare_pairs(hs_002, all_pairs, bk, mode="base")

    # ============================================================
    # Part 1: Current N(0, 0.02) with COSET mode
    # ============================================================
    log("\nPart 1: Current N(0, 0.02) embeddings - COSET mode")
    log("-" * 60)
    compare_pairs(hs_002, all_pairs, bk, mode="coset")
    del hs_002

    # ============================================================
    # Part 2: N(0, 1) embeddings - generate HS
    # ============================================================
    log("\nPart 2: Generating HS with N(0, 1) embeddings...")
    log("-" * 60)
    torch.manual_seed(42)
    emb_n01 = torch.normal(mean=0.0, std=1.0, size=(2000, 1024, 4096))
    log(f"N(0,1) embeddings generated: shape={emb_n01.shape}")

    hs_n01 = {}
    for name, path in models_info.items():
        log(f"\n  Loading {name} ({path})...")
        t0 = time.time()
        hs_n01[name] = run_model(path, emb_n01)
        log(f"  {name} done in {time.time()-t0:.0f}s, shape={hs_n01[name].shape}")

    del emb_n01

    log("\nPart 2a: N(0, 1) embeddings - BASE mode")
    log("-" * 60)
    compare_pairs(hs_n01, all_pairs, bk, mode="base")

    log("\nPart 2b: N(0, 1) embeddings - COSET mode")
    log("-" * 60)
    compare_pairs(hs_n01, all_pairs, bk, mode="coset")

    # ============================================================
    # Part 3: Also test within-family TP with N(0,1)
    # Need instruct models too
    # ============================================================
    log("\nPart 3: Within-family TP with N(0, 1) (loading instruct models)...")
    log("-" * 60)

    instruct_info = {
        "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "Mistral-7B-v0.3-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
        "Llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    }

    torch.manual_seed(42)
    emb_n01 = torch.normal(mean=0.0, std=1.0, size=(2000, 1024, 4096))

    for name, path in instruct_info.items():
        log(f"\n  Loading {name} ({path})...")
        t0 = time.time()
        hs_n01[name] = run_model(path, emb_n01)
        log(f"  {name} done in {time.time()-t0:.0f}s, shape={hs_n01[name].shape}")

    del emb_n01

    tp_pairs = [
        ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"),
        ("Mistral-7B-v0.3", "Mistral-7B-v0.3-Instruct"),
        ("Llama-2-7b", "Llama-2-7b-chat"),
    ]

    log("\nWithin-family TP - N(0,1) BASE mode:")
    compare_pairs(hs_n01, tp_pairs, bk, mode="base")

    log("\nWithin-family TP - N(0,1) COSET mode:")
    compare_pairs(hs_n01, tp_pairs, bk, mode="coset")

    # ============================================================
    # Summary
    # ============================================================
    log("\n" + "=" * 80)
    log("DONE. Check results_crossfamily.txt for full results.")


if __name__ == "__main__":
    main()
