"""
seedprint.py - Core SeedPrint fingerprinting algorithm.

Model-agnostic: operates on pre-computed hidden state tensors [N, D].
All inference and I/O utilities are in utils.py.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, norm as _norm_dist


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


# ── Main Test ────────────────────────────────────────────────────────────────

def run_test(hs_base, hs_target, buffer_k=DEFAULT_BUFFER_K,
             identity_mode=DEFAULT_IDENTITY_MODE, null_params=None,
             use_agg=False):
    """
    SeedPrint hypothesis test (z-score, analytical null).

    Primary signal (always computed):
      perdim: per-column Kendall tau on softmax_T10-normalized bottom-k
              dimensions → mean of k taus → z-score.

    Optional signal (use_agg=True):
      agg: per-sample mean of raw bottom-k values → single Kendall tau → z-score.
      Combined via max(z_perdim, z_agg) with Bonferroni correction (×2).

      Including agg raises the p-value of borderline false positives (through
      Bonferroni), while preserving detection power for true positives whose
      agg signal is also strong. Default is perdim-only (no Bonferroni needed).

    Args:
        hs_base: [N, D] base model hidden states.
        hs_target: [N, D] target model hidden states.
        buffer_k: number of bottom identity dimensions.
        identity_mode: "coset" (intersection of both models' bottom-k, default)
                       or "base" (use base model's bottom-k only).
        null_params: pre-computed from get_null_params(), or None (auto).
        use_agg: if True, include agg signal with Bonferroni correction.

    Returns:
        dict with z_perdim, p_value, k, and optionally z_agg, z_max.
    """
    # Identity positions (bottom-k)
    _, bot_A = get_identity_indices(hs_base, buffer_k)
    if identity_mode == "coset":
        _, bot_B = get_identity_indices(hs_target, buffer_k)
        idx = sorted(get_co_positions(bot_A, bot_B))
    else:
        idx = sorted(set(bot_A))
    k = len(idx)
    n = hs_base.shape[0]

    # Signal 1: per-dim (softmax_T10 → per-column Kendall tau → mean)
    A_norm = normalize_features(hs_base[:, idx], "softmax_T10")
    B_norm = normalize_features(hs_target[:, idx], "softmax_T10")
    perdim_corr, _ = compute_per_dim_correlation(A_norm, B_norm)
    perdim_mean = perdim_corr.mean().item()

    # Null parameters
    if null_params is None:
        null_params = get_null_params(n, k)

    z_perdim = (perdim_mean - null_params["perdim_mu"]) / max(null_params["perdim_std"], 1e-10)

    result = {
        "perdim_mean": perdim_mean,
        "z_perdim": z_perdim,
        "k": k,
        "n": n,
    }

    if use_agg:
        # Signal 2: agg (per-sample mean of raw bottom-k → single Kendall tau)
        agg_a = hs_base[:, idx].mean(dim=1).numpy()
        agg_b = hs_target[:, idx].mean(dim=1).numpy()
        agg_tau, _ = kendalltau(agg_a, agg_b)
        z_agg = (agg_tau - null_params["agg_mu"]) / max(null_params["agg_std"], 1e-10)

        # Combined: max z-score with Bonferroni correction for 2 tests
        z_max = max(z_perdim, z_agg)
        p_value = min(1.0, 2.0 * 2.0 * (1.0 - _norm_dist.cdf(z_max)))

        result.update({
            "agg_tau": agg_tau,
            "z_agg": z_agg,
            "z_max": z_max,
            "p_value": p_value,
        })
    else:
        # Perdim only (no Bonferroni needed)
        p_value = min(1.0, 2.0 * (1.0 - _norm_dist.cdf(z_perdim)))
        result["p_value"] = p_value

    return result
