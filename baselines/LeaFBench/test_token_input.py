"""
Quick validation: random token-ID sequences (going through embedding layer).
Compares with old embedding approach on same D=4096 pairs.
"""
import sys, os, torch, numpy as np, gc, logging
from scipy.stats import kendalltau
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fingerprint.seed.get_random_tokens import get_random_tokens
from fingerprint.seed.get_outputs import get_output_for_tokens
from benchmark.model_pool import ModelPool

NUM_SEQ   = 2000
SEQ_LEN   = 1024
MIN_VOCAB = 32000
TOKEN_DIR = "cache/seed/fingerprint_input"
CACHE_PATH = "cache/seed_token/seed_fingerprints_token.pth"
BATCH_SIZE = 4
BK_RATIO   = 0.1

MODEL_PATHS = {
    "Llama-3.1-8B":              "meta-llama/Llama-3.1-8B",
    "Llama-3.1-8B-Instruct":     "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-ExtraMix":     "Xiaojian9992024/Llama3.1-8B-ExtraMix",
    "Llama-3.1-8B-Carballo":     "proxectonos/Llama-3.1-Carballo",
    "Mistral-7B-v0.3":           "mistralai/Mistral-7B-v0.3",
    "Mistral-7B-v0.3-Instruct":  "mistralai/Mistral-7B-Instruct-v0.3",
    "Mistral-7B-v0.3-Spellcheck":"openfoodfacts/spellcheck-mistral-7b",
    "Llama-2-7b":                "meta-llama/Llama-2-7b-hf",
    "Llama-2-7b-chat":           "meta-llama/Llama-2-7b-chat-hf",
    "tulu-2-7b":                 "allenai/tulu-2-7b",
}

FAMILIES = {
    "Llama-3.1-8B":    ["Llama-3.1-8B","Llama-3.1-8B-Instruct",
                         "Llama-3.1-8B-ExtraMix","Llama-3.1-8B-Carballo","Llama-3.1-8B-TLDR"],
    "Mistral-7B-v0.3": ["Mistral-7B-v0.3","Mistral-7B-v0.3-Instruct",
                         "Mistral-7B-v0.3-Spellcheck","Mistral-7B-v0.3-Brain","AQUA-7B"],
    "Llama-2-7b":      ["Llama-2-7b","Llama-2-7b-chat","tulu-2-7b"],
}
BASE_MODELS = ["Llama-3.1-8B","Mistral-7B-v0.3","Llama-2-7b"]

def get_family(n):
    for f,ms in FAMILIES.items():
        if n in ms: return f
    return None

def bot_idx(hs, bk):
    return sorted(torch.topk(hs.mean(0), bk, largest=False).indices.tolist())

def perdim_corr(a, b, idx, T=10, max_dims=50):
    idx = sorted(idx)
    an = F.softmax(a[:,idx]/T, dim=1)
    bn = F.softmax(b[:,idx]/T, dim=1)
    taus = [kendalltau(an[:,i].numpy(), bn[:,i].numpy())[0]
            for i in range(min(max_dims, len(idx)))]
    taus = [t for t in taus if not np.isnan(t)]
    return float(np.mean(taus)) if taus else 0.0

def agg_corr(a, b, idx):
    t, _ = kendalltau(a[:,idx].mean(1).numpy(), b[:,idx].mean(1).numpy())
    return float(t) if not np.isnan(t) else 0.0

def main():
    token_path = get_random_tokens(TOKEN_DIR, NUM_SEQ, SEQ_LEN, MIN_VOCAB)

    fingerprints = {}
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached token fingerprints from {CACHE_PATH}")
        fingerprints = torch.load(CACHE_PATH, map_location='cpu')

    pool = ModelPool(max_loaded_models=1)
    log  = logging.getLogger()
    for name, path in MODEL_PATHS.items():
        if name in fingerprints:
            print(f"[SKIP] {name}")
            continue
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        pool.register_model(name, path)
        model = pool.get_model(name)
        hs = get_output_for_tokens(model, token_path, batch_size=BATCH_SIZE)
        fingerprints[name] = hs.float()
        torch.save(fingerprints, CACHE_PATH)
        print(f"Saved {name}  shape={hs.shape}")
        pool._completely_unload_model(model, name, log)
        del model; gc.collect(); torch.cuda.empty_cache()

    old_fp = {}
    old_fp_path = "cache/seed/seed_fingerprints.pth"
    if os.path.exists(old_fp_path):
        old_fp = {k: v.float() for k,v in
                  torch.load(old_fp_path, map_location='cpu').items()}

    print("\n" + "="*80)
    print("RESULTS  (token-input vs embed-input, base mode, ratio_k=0.1)")
    print("="*80)
    rows = []
    for base in BASE_MODELS:
        if base not in fingerprints: continue
        a_t = fingerprints[base]
        a_e = old_fp.get(base)
        bk  = int(BK_RATIO * a_t.shape[1])
        idx_t = bot_idx(a_t, bk)
        idx_e = bot_idx(a_e, bk) if a_e is not None else None
        print(f"\nBase: {base}  bk={bk}")
        print(f"  {'Lbl':<4} {'tok_pd':>8} {'tok_ag':>8} {'emb_pd':>8} {'emb_ag':>8}  target")
        for tgt in [m for m in fingerprints if m != base and fingerprints[m].shape[1] == a_t.shape[1]]:
            b_t  = fingerprints[tgt]
            b_e  = old_fp.get(tgt)
            lbl  = "Y" if get_family(tgt)==get_family(base) else "N"
            tp   = perdim_corr(a_t, b_t, idx_t)
            ta   = agg_corr(a_t, b_t, idx_t)
            ep   = perdim_corr(a_e, b_e, idx_e) if (a_e is not None and b_e is not None) else float('nan')
            ea   = agg_corr(a_e, b_e, idx_e)    if (a_e is not None and b_e is not None) else float('nan')
            print(f"  {lbl:<4} {tp:>8.4f} {ta:>8.4f} {ep:>8.4f} {ea:>8.4f}  {tgt}")
            rows.append((base, tgt, lbl, tp, ta, ep, ea))

    with open("results_token_input.txt","w") as f:
        f.write("base\ttarget\tlabel\ttok_pd\ttok_ag\temb_pd\temb_ag\n")
        for r in rows:
            f.write("\t".join([r[0],r[1],r[2]]+[f"{x:.4f}" for x in r[3:]])+"\n")
    print(f"\nSaved {len(rows)} rows → results_token_input.txt")

if __name__ == "__main__":
    main()
