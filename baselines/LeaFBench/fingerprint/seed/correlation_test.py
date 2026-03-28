import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import kendalltau, norm as _norm_dist
from multiprocessing import Pool, cpu_count


def get_identity_indices(hidden_states, buffer_k):
    """Return bottom-k indices from the mean of hidden_states [N, D]."""
    avg = hidden_states.mean(dim=0)  # [D]
    bot_ids = torch.topk(avg, buffer_k, largest=False).indices.tolist()
    return bot_ids


def normalize_features(tensor, method="softmax_T10"):
    if method == "softmax":
        return F.softmax(tensor, dim=1)
    elif method.startswith("softmax_T"):
        T = float(method.split("T")[1])
        return F.softmax(tensor / T, dim=1)
    else:
        return tensor


def _kendalltau_column(args):
    col_a, col_b = args
    tau, p = kendalltau(col_a, col_b)
    return tau, p


def compute_per_token_correlation(matrix_a, matrix_b):
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

    correlations = torch.tensor([r[0] for r in results])
    return correlations


def _kendall_tau_null_std(n):
    """Analytical standard deviation of Kendall tau under H0 (independence)."""
    return np.sqrt(2.0 * (2 * n + 5) / (9.0 * n * (n - 1)))


def test_lineage(base_fingerprint, testing_fingerprint, buffer_k,
                 normalize="softmax_T10", num_trials=10,
                 identity_mode="base", null_stats=None,
                 use_agg=False):
    """
    Z-score SeedPrint test.

    Primary signal (always computed):
      per-dim: per-column Kendall tau with softmax_T10 → mean of k taus

    Optional signal (use_agg=True):
      agg-bottom: per-token mean of raw bottom-k values → 1 Kendall tau
      Combined via max z-score with Bonferroni correction.

    Returns:
        p_value (float)
    """
    base_fingerprint = base_fingerprint.cpu().float()
    testing_fingerprint = testing_fingerprint.cpu().float()

    # Drop NaN rows
    valid = ~(base_fingerprint.isnan().any(dim=1) | testing_fingerprint.isnan().any(dim=1))
    base_fingerprint = base_fingerprint[valid]
    testing_fingerprint = testing_fingerprint[valid]

    n = base_fingerprint.shape[0]

    # Identity positions (base mode, bottom only)
    bot_ids = get_identity_indices(base_fingerprint, buffer_k)
    idx = sorted(bot_ids)
    k = len(idx)

    if k == 0:
        return 1.0

    # Sub-signal 1: per-dim (softmax_T10)
    A_norm = normalize_features(base_fingerprint[:, idx], normalize)
    B_norm = normalize_features(testing_fingerprint[:, idx], normalize)
    perdim_corr = compute_per_token_correlation(A_norm, B_norm)

    # Filter NaN
    valid_mask = ~torch.isnan(perdim_corr)
    perdim_corr = perdim_corr[valid_mask]
    if len(perdim_corr) == 0:
        return 1.0
    perdim_mean = perdim_corr.mean().item()
    k_valid = len(perdim_corr)

    # Analytical null
    tau_std = _kendall_tau_null_std(n)
    perdim_null_std = tau_std / np.sqrt(k_valid)

    # Z-score for perdim
    z_perdim = perdim_mean / max(perdim_null_std, 1e-10)

    if use_agg:
        # Sub-signal 2: agg-bottom (raw, no softmax)
        agg_a = base_fingerprint[:, idx].mean(dim=1).numpy()
        agg_b = testing_fingerprint[:, idx].mean(dim=1).numpy()
        agg_tau, _ = kendalltau(agg_a, agg_b)
        agg_null_std = tau_std
        z_agg = agg_tau / max(agg_null_std, 1e-10)

        # Combined: max z-score with Bonferroni correction
        z_max = max(z_perdim, z_agg)
        p_value = min(1.0, 2.0 * (1.0 - _norm_dist.cdf(z_max)))
    else:
        # Perdim-only
        p_value = min(1.0, 2.0 * (1.0 - _norm_dist.cdf(z_perdim)))

    if np.isnan(p_value):
        return 1.0

    return float(p_value)
