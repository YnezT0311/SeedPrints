"""
Full benchmark: random token-ID sequences (input_ids, through embedding layer).
All 6 families × all models.  Caches to cache/seed_token/seed_fingerprints_token.pth.
"""
import sys, os, torch, numpy as np, gc, logging
from scipy.stats import kendalltau
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fingerprint.seed.get_random_tokens import get_random_tokens
from fingerprint.seed.get_outputs import get_output_for_tokens
from benchmark.model_pool import ModelPool

NUM_SEQ    = 2000
SEQ_LEN    = 1024
MIN_VOCAB  = 32000
TOKEN_DIR  = "cache/seed/fingerprint_input"
CACHE_PATH = "cache/seed_token/seed_fingerprints_token.pth"
BATCH_SIZE = 4
BK_RATIO   = 0.1

MODEL_PATHS = {
    # Qwen-2.5-7B family
    "Qwen-2.5-7B":                    "Qwen/Qwen2.5-7B",
    "Qwen-2.5-7B-Instruct":           "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-Math":                "Qwen/Qwen2.5-Math-7B",
    "Qwen2.5-7B-Coder":               "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen2.5-7B-Instruct-Medicine":   "WangCa/Qwen2.5-7B-Medicine",
    "Qwen2.5-7B-Instruct-Abilierated":"huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2",
    "Qwen2.5-7B-Stock":               "Locutusque/StockQwen-2.5-7B",
    "QevaCoT-7B":                     "bunnycore/QevaCoT-7B-Stock",
    "Qwen2.5-7B-Instruct-Task-10":    "fangcaotank/task-10-Qwen-Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-Instruct-Task-12":    "SeeFlock/task-12-Qwen-Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-Instruct-Int4":       "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "Qwen2.5-7B-Instruct-Int8":       "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
    "Qwen2.5-7B-Open-R1-Distill":     "Lansechen/Qwen2.5-7B-Open-R1-Distill",
    # Qwen2.5-14B family
    "Qwen2.5-14B":                    "Qwen/Qwen2.5-14B",
    "Qwen2.5-14B-Instruct":           "Qwen/Qwen2.5-14B-Instruct",
    "Qwen2.5-Coder-14B":              "Qwen/Qwen2.5-Coder-14B",
    "oxy-1-small":                    "oxyapi/oxy-1-small",
    "Qwen2.5-14B-Gutenberg-Instruct-Slerpeno": "v000000/Qwen2.5-14B-Gutenberg-Instruct-Slerpeno",
    "Qwen-story-test-qlora":          "ToastyPigeon/qwen-story-test-qlora",
    "Qwen2.5-14B-Instruct-GPTQ-Int4": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
    "DeepSeek-R1-Distill-Qwen-14B":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    # Llama-3.1-8B family
    "Llama-3.1-8B":                   "meta-llama/Llama-3.1-8B",
    "Llama-3.1-8B-Instruct":          "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-Fireplace2":        "ValiantLabs/Llama3.1-8B-Fireplace2",
    "Llama-3.1-8B-TLDR":              "RedHatAI/Llama-3.1-8B-tldr",
    "Llama-3.1-8B-Carballo":          "proxectonos/Llama-3.1-Carballo",
    "Llama-3.1-8B-Instruct-Abliterated": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    "Llama-3.1-8B-Instruct-HalfAbliterated-TIES": "gaverfraxz/Meta-Llama-3.1-8B-Instruct-HalfAbliterated-TIES",
    "Llama-3.1-8B-ExtraMix":          "Xiaojian9992024/Llama3.1-8B-ExtraMix",
    "Llama-3.1-8B-Instruct-cv-job-description-matching": "LlamaFactoryAI/Llama-3.1-8B-Instruct-cv-job-description-matching",
    "Llama-3.1-8B-Instruct-PsyCourse-fold7": "chchen/Llama-3.1-8B-Instruct-PsyCourse-fold7",
    "Llama-3.1-8B-Instruct-8bit":     "iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8",
    "Llama-3.1-8B-Instruct-4bit":     "DaraV/LLaMA-3.1-8B-Instruct-INT4-GPTQ",
    "Llama-3.1-8B-Instruct-Open-R1-Distill": "asas-ai/Llama-3.1-8B-Instruct-Open-R1-Distill",
    # Mistral-7B-v0.3 family
    "Mistral-7B-v0.3":                "mistralai/Mistral-7B-v0.3",
    "Mistral-7B-v0.3-Instruct":       "mistralai/Mistral-7B-Instruct-v0.3",
    "AQUA-7B":                        "KurmaAI/AQUA-7B",
    "Mistral-7B-v0.3-Spellcheck":     "openfoodfacts/spellcheck-mistral-7b",
    "Mistral-7B-v0.3-Instruct-demi-merge": "grimjim/Mistral-7B-Instruct-demi-merge-v0.3-7B",
    "Mistral-7B-v0.3-Brain":          "chaymaemerhrioui/mistral-Brain_Model_ACC_Trainer",
    "Mistral-7B-v0.3-Instruct-GPTQ-4bit": "RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit",
    "Mistral-7B-distilled-from-deepseek-r1-qwen32b": "eganwo/mistral7b-distilled-from-deepseek-r1-qwen32b",
    # Gemma-2-2b family
    "Gemma-2-2b":                     "google/gemma-2-2b",
    "Gemma-2-2b-it":                  "google/gemma-2-2b-it",
    "Gemma-2-baku-2b":                "rinna/gemma-2-baku-2b",
    "Gemma-2-2b-neogenesis-ita":      "anakin87/gemma-2-2b-neogenesis-ita",
    "Gemma-2-2b-merged":              "vonjack/gemma2-2b-merged",
    "Gemma-2-2b-it-lora-sql":         "google-cloud-partnership/gemma-2-2b-it-lora-sql",
    "Gemma-2-2B-it-4Bit-GPTQ":        "qilowoq/gemma-2-2B-it-4Bit-GPTQ",
    "Gemma-2-2b-it-distilled":        "Syed-Hasan-8503/Gemma-2-2b-it-distilled",
    # Llama-2-7b family
    "Llama-2-7b":                     "meta-llama/Llama-2-7b-hf",
    "Llama-2-7b-chat":                "meta-llama/Llama-2-7b-chat-hf",
    "tulu-2-7b":                      "allenai/tulu-2-7b",
    "EYE-Llama_qa":                   "QIAIUNCC/EYE-Llama_qa",
    "coma-7B-v0.1":                   "DevQuasar/coma-7B-v0.1",
    "llama2-Better-Tune":             "Ammar-1/llama2-Better-Tune",
    "Llama-2-7B-Chat-GPTQ":           "TheBloke/Llama-2-7B-Chat-GPTQ",
    "llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2":
                                      "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2",
}

FAMILIES = {
    "Qwen-2.5-7B":    ["Qwen-2.5-7B","Qwen-2.5-7B-Instruct","Qwen2.5-7B-Math",
                        "Qwen2.5-7B-Coder","Qwen2.5-7B-Instruct-Medicine",
                        "Qwen2.5-7B-Instruct-Abilierated","Qwen2.5-7B-Stock",
                        "QevaCoT-7B","Qwen2.5-7B-Instruct-Task-10",
                        "Qwen2.5-7B-Instruct-Task-12","Qwen2.5-7B-Instruct-Int4",
                        "Qwen2.5-7B-Instruct-Int8","Qwen2.5-7B-Open-R1-Distill"],
    "Qwen2.5-14B":    ["Qwen2.5-14B","Qwen2.5-14B-Instruct","Qwen2.5-Coder-14B",
                        "oxy-1-small","Qwen2.5-14B-Gutenberg-Instruct-Slerpeno",
                        "Qwen-story-test-qlora","Qwen2.5-14B-Instruct-GPTQ-Int4",
                        "DeepSeek-R1-Distill-Qwen-14B"],
    "Llama-3.1-8B":   ["Llama-3.1-8B","Llama-3.1-8B-Instruct","Llama-3.1-8B-Fireplace2",
                        "Llama-3.1-8B-TLDR","Llama-3.1-8B-Carballo",
                        "Llama-3.1-8B-Instruct-Abliterated",
                        "Llama-3.1-8B-Instruct-HalfAbliterated-TIES",
                        "Llama-3.1-8B-ExtraMix",
                        "Llama-3.1-8B-Instruct-cv-job-description-matching",
                        "Llama-3.1-8B-Instruct-PsyCourse-fold7",
                        "Llama-3.1-8B-Instruct-8bit","Llama-3.1-8B-Instruct-4bit",
                        "Llama-3.1-8B-Instruct-Open-R1-Distill"],
    "Mistral-7B-v0.3":["Mistral-7B-v0.3","Mistral-7B-v0.3-Instruct","AQUA-7B",
                        "Mistral-7B-v0.3-Spellcheck","Mistral-7B-v0.3-Instruct-demi-merge",
                        "Mistral-7B-v0.3-Brain","Mistral-7B-v0.3-Instruct-GPTQ-4bit",
                        "Mistral-7B-distilled-from-deepseek-r1-qwen32b"],
    "Gemma-2-2b":     ["Gemma-2-2b","Gemma-2-2b-it","Gemma-2-baku-2b",
                        "Gemma-2-2b-neogenesis-ita","Gemma-2-2b-merged",
                        "Gemma-2-2b-it-lora-sql","Gemma-2-2B-it-4Bit-GPTQ",
                        "Gemma-2-2b-it-distilled"],
    "Llama-2-7b":     ["Llama-2-7b","Llama-2-7b-chat","tulu-2-7b","EYE-Llama_qa",
                        "coma-7B-v0.1","llama2-Better-Tune","Llama-2-7B-Chat-GPTQ",
                        "llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2"],
}

BASE_MODELS = ["Qwen-2.5-7B","Qwen2.5-14B","Llama-3.1-8B",
               "Mistral-7B-v0.3","Gemma-2-2b","Llama-2-7b"]

def get_family(n):
    for f, ms in FAMILIES.items():
        if n in ms: return f
    return None

def bot_idx(hs, bk):
    return sorted(torch.topk(hs.mean(0), bk, largest=False).indices.tolist())

def perdim_corr(a, b, idx, T=10, max_dims=50):
    idx = sorted(idx)
    an = F.softmax(a[:, idx] / T, dim=1)
    bn = F.softmax(b[:, idx] / T, dim=1)
    taus = [kendalltau(an[:, i].numpy(), bn[:, i].numpy())[0]
            for i in range(min(max_dims, len(idx)))]
    return float(np.nanmean(taus)) if taus else 0.0

def agg_corr(a, b, idx):
    t, _ = kendalltau(a[:, idx].mean(1).numpy(), b[:, idx].mean(1).numpy())
    return float(t) if not np.isnan(t) else 0.0

def main():
    token_path = get_random_tokens(TOKEN_DIR, NUM_SEQ, SEQ_LEN, MIN_VOCAB)

    fingerprints = {}
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached fingerprints from {CACHE_PATH}")
        fingerprints = torch.load(CACHE_PATH, map_location='cpu')
        # drop any NaN entries from previous failed runs
        bad = [k for k, v in fingerprints.items() if torch.is_tensor(v) and v.isnan().any()]
        for k in bad:
            print(f"  Dropping NaN entry: {k}")
            del fingerprints[k]

    pool = ModelPool(max_loaded_models=1)
    log  = logging.getLogger()
    for name, path in MODEL_PATHS.items():
        if name in fingerprints:
            print(f"[SKIP] {name}")
            continue
        print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
        pool.register_model(name, path)
        try:
            model = pool.get_model(name)
            hs = get_output_for_tokens(model, token_path, batch_size=BATCH_SIZE)
            if hs.isnan().any():
                print(f"  WARNING: NaN in output for {name}, skipping")
            else:
                fingerprints[name] = hs.float()
                torch.save(fingerprints, CACHE_PATH)
                print(f"Saved {name}  shape={hs.shape}", flush=True)
            pool._completely_unload_model(model, name, log)
            del model
        except Exception as e:
            print(f"  ERROR: {e}")
        gc.collect(); torch.cuda.empty_cache()

    # ── correlation analysis ──────────────────────────────────────────────
    old_fp = {}
    if os.path.exists("cache/seed/seed_fingerprints.pth"):
        old_fp = {k: v.float() for k, v in
                  torch.load("cache/seed/seed_fingerprints.pth", map_location='cpu').items()}

    print("\n" + "="*100)
    print("RESULTS  (token-input, ratio_k=0.1)")
    print("="*100)
    rows = []
    for base in BASE_MODELS:
        if base not in fingerprints: continue
        a_t = fingerprints[base]; a_e = old_fp.get(base)
        bk  = int(BK_RATIO * a_t.shape[1])
        idx_t = bot_idx(a_t, bk)
        idx_e = bot_idx(a_e, bk) if a_e is not None else None
        cands = sorted([m for m in fingerprints
                        if m != base and fingerprints[m].shape[1] == a_t.shape[1]])
        print(f"\nBase: {base}  D={a_t.shape[1]}  bk={bk}  ({len(cands)} candidates)")
        print(f"  {'Same':>5} {'tok_pd':>8} {'tok_ag':>8} {'emb_pd':>8} {'emb_ag':>8}  target")
        for tgt in cands:
            b_t = fingerprints[tgt]; b_e = old_fp.get(tgt)
            lbl = "Y" if get_family(tgt) == get_family(base) else "N"
            tp  = perdim_corr(a_t, b_t, idx_t)
            ta  = agg_corr(a_t, b_t, idx_t)
            ep  = perdim_corr(a_e, b_e, idx_e) if a_e is not None and b_e is not None else float('nan')
            ea  = agg_corr(a_e, b_e, idx_e)    if a_e is not None and b_e is not None else float('nan')
            print(f"  {lbl:>5} {tp:>8.4f} {ta:>8.4f} {ep:>8.4f} {ea:>8.4f}  {tgt}")
            rows.append((base, tgt, lbl, tp, ta, ep, ea))

    with open("results_token_input_full.tsv", "w") as f:
        f.write("base\ttarget\tsame_origin\ttok_pd\ttok_ag\temb_pd\temb_ag\n")
        for r in rows:
            f.write("\t".join([r[0], r[1], r[2]] + [f"{x:.4f}" for x in r[3:]]) + "\n")
    print(f"\nSaved {len(rows)} rows → results_token_input_full.tsv")

if __name__ == "__main__":
    main()
