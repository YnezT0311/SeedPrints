"""
Compute per-pair correlation table for all same-D pairs in the benchmark.
Outputs base_perdim, base_agg, coset_perdim, coset_agg + TP/FP label.
"""
import torch
import numpy as np
from scipy.stats import kendalltau
import torch.nn.functional as F

BK_RATIO = 0.1  # ratio_k

FAMILIES = {
    "Qwen-2.5-7B":    ["Qwen-2.5-7B", "Qwen-2.5-7B-Instruct", "Qwen2.5-7B-Math",
                        "Qwen2.5-7B-Coder", "Qwen2.5-7B-Instruct-Medicine",
                        "Qwen2.5-7B-Instruct-Abilierated", "Qwen2.5-7B-Stock",
                        "QevaCoT-7B", "Qwen2.5-7B-Instruct-Task-10",
                        "Qwen2.5-7B-Instruct-Task-12", "Qwen2.5-7B-Instruct-Int4",
                        "Qwen2.5-7B-Instruct-Int8", "Qwen2.5-7B-Open-R1-Distill"],
    "Qwen2.5-14B":    ["Qwen2.5-14B", "Qwen2.5-14B-Instruct", "Qwen2.5-Coder-14B",
                        "oxy-1-small", "Qwen2.5-14B-Gutenberg-Instruct-Slerpeno",
                        "Qwen-story-test-qlora", "Qwen2.5-14B-Instruct-GPTQ-Int4",
                        "DeepSeek-R1-Distill-Qwen-14B"],
    "Llama-3.1-8B":   ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-8B-Fireplace2",
                        "Llama-3.1-8B-TLDR", "Llama-3.1-8B-Carballo",
                        "Llama-3.1-8B-Instruct-Abliterated",
                        "Llama-3.1-8B-Instruct-HalfAbliterated-TIES",
                        "Llama-3.1-8B-ExtraMix",
                        "Llama-3.1-8B-Instruct-cv-job-description-matching",
                        "Llama-3.1-8B-Instruct-PsyCourse-fold7",
                        "Llama-3.1-8B-Instruct-8bit", "Llama-3.1-8B-Instruct-4bit",
                        "Llama-3.1-8B-Instruct-Open-R1-Distill"],
    "Mistral-7B-v0.3": ["Mistral-7B-v0.3", "Mistral-7B-v0.3-Instruct", "AQUA-7B",
                         "Mistral-7B-v0.3-Spellcheck", "Mistral-7B-v0.3-Instruct-demi-merge",
                         "Mistral-7B-v0.3-Brain", "Mistral-7B-v0.3-Instruct-GPTQ-4bit",
                         "Mistral-7B-distilled-from-deepseek-r1-qwen32b"],
    "Gemma-2-2b":     ["Gemma-2-2b", "Gemma-2-2b-it", "Gemma-2-baku-2b",
                        "Gemma-2-2b-neogenesis-ita", "Gemma-2-2b-merged",
                        "Gemma-2-2b-it-lora-sql", "Gemma-2-2B-it-4Bit-GPTQ",
                        "Gemma-2-2b-it-distilled"],
    "Llama-2-7b":     ["Llama-2-7b", "Llama-2-7b-chat", "tulu-2-7b", "EYE-Llama_qa",
                        "coma-7B-v0.1", "llama2-Better-Tune", "Llama-2-7B-Chat-GPTQ",
                        "llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2"],
}

BASE_MODELS = ["Qwen-2.5-7B", "Qwen2.5-14B", "Llama-3.1-8B",
               "Mistral-7B-v0.3", "Gemma-2-2b", "Llama-2-7b"]

def get_family(name):
    for fam, members in FAMILIES.items():
        if name in members:
            return fam
    return None

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
    return t if not np.isnan(t) else 0.0

def main():
    print("Loading cached fingerprints...", flush=True)
    data = torch.load('cache/seed/seed_fingerprints.pth', map_location='cpu')
    hs = {k: v.float() for k, v in data.items()}

    # Build model-to-family lookup
    model_to_fam = {}
    for name in hs:
        model_to_fam[name] = get_family(name)

    rows = []
    for base_name in BASE_MODELS:
        if base_name not in hs:
            print(f"  WARNING: {base_name} not in cache, skip")
            continue
        a = hs[base_name]
        D = a.shape[1]
        bk = int(BK_RATIO * D)
        idx_a = get_bot_idx(a, bk)
        base_fam = get_family(base_name)

        # All models with same D (excluding base itself)
        candidates = [m for m in hs if m != base_name and hs[m].shape[1] == D]

        print(f"\nBase: {base_name} (D={D}, bk={bk}), {len(candidates)} candidates", flush=True)
        for tgt_name in candidates:
            b = hs[tgt_name]
            tgt_fam = get_family(tgt_name)
            label = "Y" if tgt_fam == base_fam else "N"

            # BASE mode
            bp = perdim_corr(a, b, idx_a)
            ba = agg_corr(a, b, idx_a)

            # COSET mode
            idx_b = get_bot_idx(b, bk)
            coset_idx = sorted(set(idx_a) & set(idx_b))
            k = len(coset_idx)
            if k == 0:
                cp, ca = float('nan'), float('nan')
            else:
                cp = perdim_corr(a, b, coset_idx)
                ca = agg_corr(a, b, coset_idx)

            rows.append({
                "base": base_name,
                "target": tgt_name,
                "base_fam": base_fam,
                "tgt_fam": tgt_fam,
                "label": label,
                "D": D,
                "bk": bk,
                "coset_k": k,
                "base_perdim": bp,
                "base_agg": ba,
                "coset_perdim": cp,
                "coset_agg": ca,
            })
            print(f"  {label} {tgt_name:50s} base_pd={bp:.4f} base_ag={ba:.4f} "
                  f"coset_pd={cp:.4f} coset_ag={ca:.4f} k={k}", flush=True)

    # Save to TSV
    import csv
    outfile = "results_corr_table.tsv"
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {outfile}")

    # Print per-base-model summary: sorted by base_perdim descending
    print("\n" + "="*100)
    print("SUMMARY (sorted by base_perdim descending per base model)")
    print("="*100)
    for base_name in BASE_MODELS:
        sub = [r for r in rows if r["base"] == base_name]
        if not sub:
            continue
        sub.sort(key=lambda r: r["base_perdim"], reverse=True)
        D = sub[0]["D"]
        print(f"\n--- Base: {base_name} (D={D}) ---")
        print(f"  {'Label':<4} {'base_pd':>8} {'base_ag':>8} {'coset_pd':>10} {'coset_ag':>10} {'k':>5}  Target")
        for r in sub:
            print(f"  {r['label']:<4} {r['base_perdim']:>8.4f} {r['base_agg']:>8.4f} "
                  f"{r['coset_perdim']:>10.4f} {r['coset_agg']:>10.4f} {r['coset_k']:>5}  {r['target']}")

if __name__ == "__main__":
    main()
