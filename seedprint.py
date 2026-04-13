"""
seedprint.py - Core SeedPrint fingerprinting algorithm.

Model-agnostic: operates on pre-computed hidden state tensors [N, D].
All inference and I/O utilities are in utils.py.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, ttest_ind, mannwhitneyu, norm as _norm_dist


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_NORMALIZE = "softmax_T10"
DEFAULT_IDENTITY_MODE = "coset"      # "coset" (intersection) or "base"
DEFAULT_BUFFER_K = 400


# ── Identity Index Extraction ────────────────────────────────────────────────

def get_identity_indices(hidden_states, buffer_k=DEFAULT_BUFFER_K):
    """
    Select identity dimensions from mean activation profile.

    Args:
        hidden_states: [N, D] tensor.
        buffer_k: number of top/bottom dimensions to select.
    Returns:
        (top_ids, bot_ids): lists of dimension indices.
    """
    avg = hidden_states.mean(dim=0)  # [D]
    top_ids = torch.topk(avg, buffer_k, largest=True).indices.tolist()
    bot_ids = torch.topk(avg, buffer_k, largest=False).indices.tolist()
    return top_ids, bot_ids


def get_co_positions(ids_a, ids_b):
    """Set intersection of two index lists."""
    return set(ids_a) & set(ids_b)


# ── Normalization ────────────────────────────────────────────────────────────

def normalize_features(tensor, method=DEFAULT_NORMALIZE):
    """
    Row-wise normalization of a [N, K] tensor.

    Supported methods:
      "none"          - passthrough
      "softmax"       - F.softmax(tensor, dim=1)
      "softmax_T{N}"  - F.softmax(tensor / N, dim=1)  (e.g. "softmax_T10")
    """
    if method == "none":
        return tensor
    if method == "softmax":
        return F.softmax(tensor, dim=1)
    if method.startswith("softmax_T"):
        T = float(method.split("T")[1])
        return F.softmax(tensor / T, dim=1)
    raise ValueError(f"Unknown normalize method: {method}")


# ── Kendall-Tau Correlation ──────────────────────────────────────────────────

def _kendalltau_column(args):
    col_a, col_b = args
    tau, p = kendalltau(col_a, col_b)
    return tau, p


def compute_per_dim_correlation(matrix_a, matrix_b):
    """
    Per-column Kendall-Tau between two [N, K] matrices.

    Parallelized when K > 100.

    Returns:
        (corr_vector [K], p_values [K])
    """
    from multiprocessing import Pool, cpu_count
    k = matrix_a.shape[1]
    a_np = matrix_a.numpy()
    b_np = matrix_b.numpy()
    args = [(a_np[:, i], b_np[:, i]) for i in range(k)]

    if k > 100:
        with Pool(min(cpu_count(), 32)) as pool:
            results = pool.map(_kendalltau_column, args)
    else:
        results = [_kendalltau_column(a) for a in args]

    correlations = [r[0] for r in results]
    p_values = [r[1] for r in results]
    return torch.tensor(correlations), torch.tensor(p_values)


# ── Analytical Null Distribution ─────────────────────────────────────────────

def _kendall_tau_null_std(n):
    """Standard deviation of Kendall tau under H0 (independence)."""
    return np.sqrt(2.0 * (2 * n + 5) / (9.0 * n * (n - 1)))


def get_null_params(n, k):
    """
    Analytical null distribution parameters.

    Under H0 (two independent random vectors):
      - Single Kendall tau: mean=0, std=sqrt(2(2n+5) / (9n(n-1)))
      - Mean of k independent taus: mean=0, std=tau_std / sqrt(k)

    Args:
        n: number of samples.
        k: number of identity dimensions.
    Returns:
        dict with null parameters for perdim and agg signals.
    """
    tau_std = _kendall_tau_null_std(n)
    return {
        "perdim_mu": 0.0,
        "perdim_std": tau_std / np.sqrt(k),
        "agg_mu": 0.0,
        "agg_std": tau_std,
        "n": n,
        "k": k,
    }


# ── Empirical Baselines ─────────────────────────────────────────────────────

def _gen_baseline_simplified(n, k, normalize=DEFAULT_NORMALIZE):
    """
    Generate one random baseline: two N(0,I) matrices [n, k] → normalize
    → per-column Kendall tau. Skips identity extraction.
    """
    A = normalize_features(torch.randn(n, k), normalize)
    B = normalize_features(torch.randn(n, k), normalize)
    corr_vec, _ = compute_per_dim_correlation(A, B)
    return corr_vec


def _gen_baseline_full_pipeline(shape, identity_mode=DEFAULT_IDENTITY_MODE,
                                normalize=DEFAULT_NORMALIZE,
                                buffer_k=DEFAULT_BUFFER_K):
    """
    Generate one random baseline through the full pipeline: two random [N, D]
    matrices → identity extraction → co-positions → normalize → per-column
    Kendall tau. Tests the entire pipeline end-to-end.
    """
    A = torch.randn(shape)
    B = torch.randn(shape)

    _, bot_A = get_identity_indices(A, buffer_k)
    if identity_mode == "coset":
        _, bot_B = get_identity_indices(B, buffer_k)
        idx = sorted(get_co_positions(bot_A, bot_B))
    else:
        idx = sorted(set(bot_A))

    if len(idx) == 0:
        return torch.zeros(1)

    co_A = normalize_features(A[:, idx], normalize)
    co_B = normalize_features(B[:, idx], normalize)
    corr_vec, _ = compute_per_dim_correlation(co_A, co_B)
    return corr_vec


# ── Main Test ────────────────────────────────────────────────────────────────

def run_test(hs_base, hs_target, buffer_k=DEFAULT_BUFFER_K,
             identity_mode=DEFAULT_IDENTITY_MODE, use_agg=False,
             method="analytical", num_trials=10, test_type="t-test",
             baseline="simplified", null_params=None):
    """
    SeedPrint hypothesis test.

    Three testing methods:
      - "analytical" (default): z-score against closed-form null distribution.
        Assumes per-column Kendall taus are independent (approximately true
        for high-temperature softmax). Fastest and deterministic.
      - "empirical" with baseline="simplified": compare against null baselines
        generated from random N(0,I) [n,k] matrices (same normalization, no
        identity extraction). Captures softmax-induced cross-column dependencies.
      - "empirical" with baseline="full_pipeline": compare against null baselines
        generated through the entire pipeline (random [N,D] → identity extraction
        → normalize → correlate). Most thorough; also validates the analytical
        approximation.

    Primary signal (always computed):
      perdim: per-column Kendall tau on softmax_T10-normalized bottom-k
              dimensions → mean of k taus.

    Optional signal (use_agg=True):
      agg: per-sample mean of raw bottom-k values → single Kendall tau.
      Combined via max z-score with Bonferroni correction (×2).

    Args:
        hs_base: [N, D] base model hidden states.
        hs_target: [N, D] target model hidden states.
        buffer_k: number of bottom identity dimensions.
        identity_mode: "coset" (default) or "base".
        use_agg: if True, include agg signal with Bonferroni correction.
        method: "analytical" (default) or "empirical".
        num_trials: number of random baseline trials (empirical only).
        test_type: "t-test" (default) or "u-test" (empirical only).
        baseline: "simplified" or "full_pipeline" (empirical only).
        null_params: pre-computed from get_null_params() (analytical only).

    Returns:
        dict with perdim_mean, z_perdim (or t/u stats), p_value, k, n.
    """
    # ── Identity positions (bottom-k) ──
    _, bot_A = get_identity_indices(hs_base, buffer_k)
    if identity_mode == "coset":
        _, bot_B = get_identity_indices(hs_target, buffer_k)
        idx = sorted(get_co_positions(bot_A, bot_B))
    else:
        idx = sorted(set(bot_A))
    k = len(idx)
    n = hs_base.shape[0]

    # ── Signal: per-dim correlation ──
    A_norm = normalize_features(hs_base[:, idx], "softmax_T10")
    B_norm = normalize_features(hs_target[:, idx], "softmax_T10")
    perdim_corr, _ = compute_per_dim_correlation(A_norm, B_norm)
    perdim_mean = perdim_corr.mean().item()

    result = {
        "perdim_mean": perdim_mean,
        "k": k,
        "n": n,
        "method": method,
    }

    # ── Analytical z-score ──
    if method == "analytical":
        if null_params is None:
            null_params = get_null_params(n, k)

        z_perdim = (perdim_mean - null_params["perdim_mu"]) / max(null_params["perdim_std"], 1e-10)
        result["z_perdim"] = z_perdim

        if use_agg:
            agg_a = hs_base[:, idx].mean(dim=1).numpy()
            agg_b = hs_target[:, idx].mean(dim=1).numpy()
            agg_tau, _ = kendalltau(agg_a, agg_b)
            z_agg = (agg_tau - null_params["agg_mu"]) / max(null_params["agg_std"], 1e-10)
            z_max = max(z_perdim, z_agg)
            # Use logsf for numerical stability (avoids p=0 for large z)
            log_p = np.log(2) + np.log(2) + _norm_dist.logsf(z_max)
            p_value = min(1.0, np.exp(log_p))
            result.update({"agg_tau": agg_tau, "z_agg": z_agg, "z_max": z_max,
                           "log10_p": log_p / np.log(10)})
        else:
            log_p = np.log(2) + _norm_dist.logsf(z_perdim)
            p_value = min(1.0, np.exp(log_p))
            result["log10_p"] = log_p / np.log(10)

        result["p_value"] = p_value

    # ── Empirical hypothesis test ──
    elif method == "empirical":
        corr_flat = perdim_corr.numpy().flatten()

        # Generate random baselines
        baseline_corrs = []
        for _ in range(num_trials):
            if baseline == "full_pipeline":
                rcv = _gen_baseline_full_pipeline(
                    hs_base.shape, identity_mode=identity_mode,
                    normalize="softmax_T10", buffer_k=buffer_k)
            else:
                rcv = _gen_baseline_simplified(n, k, normalize="softmax_T10")
            baseline_corrs.append(rcv)

        # Pool all baseline correlations
        null_flat = torch.cat([r.flatten() for r in baseline_corrs]).numpy()

        # Statistical test (one-sided: real > null)
        if test_type == "u-test":
            stat, p_value = mannwhitneyu(corr_flat, null_flat, alternative='greater')
            result["u_stat"] = float(stat)
        else:
            stat, p_value = ttest_ind(corr_flat, null_flat, alternative='greater')
            result["t_stat"] = float(stat)

        result.update({
            "p_value": float(p_value),
            "null_mean": float(null_flat.mean()),
            "null_std": float(null_flat.std()),
            "num_trials": num_trials,
            "baseline": baseline,
            "test_type": test_type,
        })

    else:
        raise ValueError(f"Unknown method: {method}. Use 'analytical' or 'empirical'.")

    return result
