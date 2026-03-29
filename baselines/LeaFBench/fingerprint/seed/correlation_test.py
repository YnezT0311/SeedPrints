import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import kendalltau, norm as _norm_dist
from multiprocessing import Pool, cpu_count


def get_identity_indices(hidden_states, buffer_k):
    """Return bottom-k indices from the mean of hidden_states [N, D]."""
    avg = hidden_states.mean(dim=0)
    bot_ids = torch.topk(avg, buffer_k, largest=False).indices.tolist()
    return bot_ids


def normalize_features(tensor, method="softmax_T10"):
    if method == "softmax":
        return F.softmax(tensor, dim=1)
    if method.startswith("softmax_T"):
        T = float(method.split("T")[1])
        return F.softmax(tensor / T, dim=1)
    return tensor


def _kendalltau_column(args):
    col_a, col_b = args
    tau, p = kendalltau(col_a, col_b)
    return tau, p


def compute_per_dim_correlation(matrix_a, matrix_b):
    """Kendall-Tau per column, parallelized for large k."""
    k = matrix_a.shape[1]
    a_np = matrix_a.numpy() if isinstance(matrix_a, torch.Tensor) else matrix_a
    b_np = matrix_b.numpy() if isinstance(matrix_b, torch.Tensor) else matrix_b
    args = [(a_np[:, i], b_np[:, i]) for i in range(k)]

    if k > 100:
        with Pool(min(cpu_count(), 32)) as pool:
            results = pool.map(_kendalltau_column, args)
    else:
        results = [_kendalltau_column(a) for a in args]

    return torch.tensor([r[0] for r in results])


def _kendall_tau_null_std(n):
    """Analytical standard deviation of Kendall tau under H0."""
    return np.sqrt(2.0 * (2 * n + 5) / (9.0 * n * (n - 1)))


def test_lineage(base_fingerprint, testing_fingerprint, buffer_k,
                 normalize="softmax_T10", identity_mode="coset",
                 use_agg=False):
    """
    SeedPrint z-score hypothesis test.

    Primary signal (always computed):
      perdim: per-column Kendall tau on softmax_T10 bottom-k → mean → z-score

    Optional signal (use_agg=True):
      agg: per-sample mean of raw bottom-k → single Kendall tau → z-score
      Combined via max(z_perdim, z_agg) with Bonferroni correction (×2).

      Including agg raises the p-value of borderline false positives (through
      Bonferroni), while preserving power for true positives whose agg signal
      is also strong.  Default is perdim-only (no Bonferroni needed).

    Args:
        base_fingerprint: [N, D] tensor (base model hidden states).
        testing_fingerprint: [N, D] tensor (target model hidden states).
        buffer_k: number of bottom identity dimensions.
        normalize: normalization method (e.g. "softmax_T10").
        identity_mode: "coset" (intersection, default) or "base".
        use_agg: include agg signal with Bonferroni correction.

    Returns:
        p_value (float): evidence against lineage (smaller = more related).
    """
    base_fingerprint = base_fingerprint.cpu().float()
    testing_fingerprint = testing_fingerprint.cpu().float()

    # Drop NaN rows
    valid = ~(base_fingerprint.isnan().any(dim=1) | testing_fingerprint.isnan().any(dim=1))
    base_fingerprint = base_fingerprint[valid]
    testing_fingerprint = testing_fingerprint[valid]
    n = base_fingerprint.shape[0]

    # Identity positions (bottom-k)
    bot_base = get_identity_indices(base_fingerprint, buffer_k)
    if identity_mode == "coset":
        bot_target = get_identity_indices(testing_fingerprint, buffer_k)
        idx = sorted(set(bot_base) & set(bot_target))
    else:
        idx = sorted(set(bot_base))
    k = len(idx)

    if k == 0:
        return 1.0

    # Signal 1: per-dim (softmax_T10 → per-column Kendall tau → mean)
    A_norm = normalize_features(base_fingerprint[:, idx], normalize)
    B_norm = normalize_features(testing_fingerprint[:, idx], normalize)
    perdim_corr = compute_per_dim_correlation(A_norm, B_norm)

    valid_mask = ~torch.isnan(perdim_corr)
    perdim_corr = perdim_corr[valid_mask]
    if len(perdim_corr) == 0:
        return 1.0
    perdim_mean = perdim_corr.mean().item()
    k_valid = len(perdim_corr)

    # Analytical null
    tau_std = _kendall_tau_null_std(n)
    perdim_null_std = tau_std / np.sqrt(k_valid)
    z_perdim = perdim_mean / max(perdim_null_std, 1e-10)

    if use_agg:
        # Signal 2: agg (per-sample mean of raw bottom-k → single Kendall tau)
        agg_a = base_fingerprint[:, idx].mean(dim=1).numpy()
        agg_b = testing_fingerprint[:, idx].mean(dim=1).numpy()
        agg_tau, _ = kendalltau(agg_a, agg_b)
        z_agg = agg_tau / max(tau_std, 1e-10)

        # Combined: max z with Bonferroni (×2 for two-sided, ×2 for 2 tests)
        z_max = max(z_perdim, z_agg)
        p_value = min(1.0, 2.0 * 2.0 * (1.0 - _norm_dist.cdf(z_max)))
    else:
        # Perdim only (two-sided, no Bonferroni)
        p_value = min(1.0, 2.0 * (1.0 - _norm_dist.cdf(z_perdim)))

    if np.isnan(p_value):
        return 1.0

    return float(p_value)
