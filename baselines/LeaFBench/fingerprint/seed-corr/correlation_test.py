import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

from fingerprint.seed.get_outputs import get_output_for_co_indices

def _get_identity_indices(
    hidden_states: torch.Tensor,
    peak_rel_height: float = 0.15,  # 峰过滤阈值：相对 dist 最大值
    normalize_xy: bool = False):

    probs = hidden_states
    N, D = probs.shape
    avg_probs = probs.mean(dim=0)  # [D]

    # sort by avg_probs
    sorted_probs, sorted_indices = torch.sort(avg_probs)
    y = sorted_probs.detach().float().cpu()
    x = torch.arange(D, dtype=torch.float32)

    # normalization (Optional)
    eps = 1e-12
    if normalize_xy:
        x0 = (x - x.min()) / (x.max() - x.min() + eps)
        y0 = (y - y.min()) / (y.max() - y.min() + eps)
    else:
        x0=x
        y0=y

    # ===== 最大弦距 dist =====
    p1x, p1y = x0[0], y0[0]
    p2x, p2y = x0[-1], y0[-1]
    vx, vy = (p2x - p1x), (p2y - p1y)
    vnorm = torch.sqrt(vx * vx + vy * vy) + eps
    dist = torch.abs((x0 - p1x) * vy - (y0 - p1y) * vx) / vnorm  # [D]

    # ===== 找局部峰（不用任何平滑/padding）=====
    d = dist
    peak_mask = (d[1:-1] > d[0:-2]) & (d[1:-1] >= d[2:])
    peaks = torch.nonzero(peak_mask).view(-1) + 1  # 按rank从小到大

    if peaks.numel() > 0:
        thr = peak_rel_height * float(d.max().item() + eps)
        peaks = peaks[d[peaks] >= thr]

    # ===== 选 knee & 选 tokens =====
    knee_rank = int(torch.argmax(d[len(d)//2:]).item() + len(d)//2)
    selected_sorted = sorted_indices[:knee_rank + 1]


    stable_bottom_ids = selected_sorted.detach().cpu().tolist()

    return stable_bottom_ids

# def _get_fingerprint_indices(outputs_target, outputs_base):
#     avg_outputs_target = outputs_target.mean(dim=0) # [vocab_size]
#     avg_outputs_base = outputs_base.mean(dim=0) # [vocab_size]
#     # get the bottom-k positions in the avg logits
#     bottom_k_token_ids_target = torch.topk(avg_outputs_target, num_k, largest=False).indices.tolist()
#     bottom_k_token_ids_base = torch.topk(avg_outputs_base, num_k, largest=False).indices.tolist()
#     # get the co-positions between the base model and the suspected model
#     set_base = set(bottom_k_token_ids_base)
#     set_suspected = set(bottom_k_token_ids_target)
#     co_positions = set_base.intersection(set_suspected)
#     return co_positions

def _get_fingerprint_indices(num_k, outputs_target, outputs_base):
    bottom_k_token_ids_target = _get_identity_indices(outputs_target)
    bottom_k_token_ids_base = _get_identity_indices(outputs_base)
    # get the co-positions between the base model and the suspected model
    set_base = set(bottom_k_token_ids_base)
    set_suspected = set(bottom_k_token_ids_target)
    co_positions = set_base.intersection(set_suspected)
    return co_positions

def _get_probs(logits):
    """
    Convert logits to probabilities using softmax.
    
    Args:
        logits: List of tensors, each of shape [vocab_size].
    
    Returns:
        probs: List of tensors, each of shape [vocab_size].
    """
    probs = []
    for logit in logits:
        prob = F.softmax(logit, dim=-1)
        probs.append(prob)
    probs = torch.stack(probs, dim=0) # [num_samples, vocab_size]
    return probs

def _compute_per_indice_correlation(outputs_base, outputs_target):
    """Compute Kendall's Tau correlation for each token across sequences"""
    k = outputs_base.shape[1]
    correlations = []
    p_values = []
    for i in range(k):
        tau, p = kendalltau(outputs_base[:, i], outputs_target[:, i])
        correlations.append(tau)
        p_values.append(p)
    return torch.tensor(correlations), torch.tensor(p_values)

def pool(x):
    if isinstance(x, torch.Tensor):
        return x.reshape(-1).detach().cpu().numpy()
    return x.reshape(-1)

def get_corrs(base_fingerprint, testing_fingerprint, output_type, num_k, num_random_baseline_trials):
    """Compute correlation between two sets of hidden states or logits at co-positions."""
    # if any of the trial of fingerprint is NaN, delete that trial for both. fingerprint shape [trials, hidden_size]
    valid_trials = ~torch.isnan(testing_fingerprint).any(dim=1) & ~torch.isnan(base_fingerprint).any(dim=1)
    testing_fingerprint = testing_fingerprint[valid_trials]
    base_fingerprint = base_fingerprint[valid_trials]
    
    co_indices = _get_fingerprint_indices(num_k, testing_fingerprint, base_fingerprint)
    # print(f"Number of co-positions in bottom-{num_k}: {len(co_indices)}")

    co_fingerprint_target = get_output_for_co_indices(output_type, testing_fingerprint, co_indices)
    co_fingerprint_base = get_output_for_co_indices(output_type, base_fingerprint, co_indices)
    co_probs_target = _get_probs(co_fingerprint_target).cpu()
    co_probs_base = _get_probs(co_fingerprint_base).cpu()
    corr_vector, p_values = _compute_per_indice_correlation(co_probs_base, co_probs_target)
    
    # Prepare random baseline
    random_corr_vectors = []
    for _ in range(num_random_baseline_trials):
        random_corr_vectors.append(_gen_random_corr_vector_like(testing_fingerprint.shape, _compute_per_indice_correlation))
    random_pooled = torch.cat([rcv.reshape(-1) for rcv in random_corr_vectors], dim=0)
    random_pooled = pool(random_pooled)
    return corr_vector, p_values, random_pooled

def _gen_random_corr_vector_like(shape, compute_corr_fn):
    """
    probs_like: a tensor shaped like probs_target (or probs_base): [N, V] or similar
    compute_corr_fn: callable taking (X, Y) -> corr_vector, _ ; same as compute_per_token_correlation
    """
    A = torch.randn(shape)
    B = torch.randn(shape)

    # get co positions
    bottom_k_token_ids_A = _get_identity_indices(A)
    bottom_k_token_ids_B = _get_identity_indices(B)
    co_positions = set(bottom_k_token_ids_A).intersection(set(bottom_k_token_ids_B))
    co_value_A = A[:, list(co_positions)]
    co_value_B = B[:, list(co_positions)]
    A = F.softmax(co_value_A, dim=-1)
    B = F.softmax(co_value_B, dim=-1)
    corr_vec, _ = compute_corr_fn(A, B)
    return corr_vec