"""Deterministic window voting, distance supervision and sequence assembly."""
import numpy as np

WINDOW = 10
MAX_POSITIVES = 6


def windows(n):
    if n < 1:
        raise ValueError("A clip must have at least one frame")
    # Short clips have one padded window, never invented overlap support.
    return np.minimum(np.arange(max(1, n - WINDOW + 1))[:, None]
                      + np.arange(WINDOW), n - 1)


def valid_windows(n):
    ids = windows(n)
    valid = np.ones_like(ids, dtype=bool)
    if n < WINDOW:
        valid[:, n:] = False
    return ids, valid


def boundaries(n):
    return sorted({i for i in (0, 1, n - 2, n - 1) if 0 <= i < n})


def distance_targets(n, truth):
    truth = sorted(set(truth))
    if not truth:
        return np.zeros(n, np.float32)
    if min(truth) < 0 or max(truth) >= n:
        raise ValueError("Ground-truth frame outside clip")
    gt, positions = np.array(truth), np.arange(n)
    right = np.searchsorted(gt, positions).clip(max=len(gt)-1)
    left = (right-1).clip(min=0)
    distance = np.minimum(np.abs(positions-gt[left]), np.abs(positions-gt[right]))
    return np.where(distance <= 3, 1 - distance / 4, 0).astype(np.float32)


def vote(probabilities, n, threshold=0.5):
    ids, valid = valid_windows(n)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != ids.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected finite probabilities shaped [number_of_windows, 10]")
    chosen = np.zeros_like(valid)
    for w in range(len(ids)):
        rank = sorted(np.flatnonzero(valid[w]), key=lambda j: (-p[w, j], j))[:6]
        chosen[w, rank] = p[w, rank] >= threshold
    support = np.bincount(ids[valid], minlength=n)
    yes = np.bincount(ids[chosen], minlength=n)
    total = np.bincount(ids[valid], weights=p[valid], minlength=n)
    return chosen, yes / np.maximum(support, 1), total / np.maximum(support, 1), support


def split_sequences(indices, max_gap=3):
    """Group nearby positives into 2/3-frame proposals. Drop isolated positives."""
    runs = []
    for i in sorted(set(indices)):
        if not runs or i - runs[-1][-1] > max_gap:
            runs.append([i])
        else:
            runs[-1].append(i)
    result = []
    for run in runs:
        while len(run) >= 2:
            size = 2 if len(run) in (2, 4) else 3
            result.append(run[:size])
            run = run[size:]
    return result


def decode(probabilities, n, budget=None, threshold=0.5, consensus=0.5):
    chosen, agreement, confidence, support = vote(probabilities, n, threshold)
    forced = boundaries(n)
    budget = n if budget is None else max(len(forced), min(n, int(budget)))
    eligible = [i for i in range(n) if i not in forced and support[i] >= 3
                and agreement[i] >= consensus]
    eligible.sort(key=lambda i: (-agreement[i], -confidence[i], i))
    # Ranking is by overlap agreement first, then mean probability.
    proposals = split_sequences(eligible[:budget - len(forced)])
    selected = sorted(set(forced + [i for seq in proposals for i in seq]))
    return {"selected_indices": selected, "event_sequences": proposals,
            "boundary_context": split_sequences(forced, max_gap=1),
            "forced_indices": forced, "budget": budget,
            "agreement": agreement.tolist(), "support": support.tolist(),
            "unsupported_interior": [i for i in range(n) if i not in forced and support[i] < 3],
            "window_positives": chosen.astype(int).tolist()}


def match_metrics(predicted, truth, tolerance=3):
    """Maximum-cardinality order-preserving matching; no repeated GT credit."""
    p, t = sorted(set(predicted)), sorted(set(truth))
    dp = np.zeros((len(p) + 1, len(t) + 1), dtype=int)
    for i, a in enumerate(p, 1):
        for j, b in enumerate(t, 1):
            dp[i, j] = max(dp[i - 1, j], dp[i, j - 1],
                           dp[i - 1, j - 1] + int(abs(a - b) <= tolerance))
    tp = int(dp[-1, -1])
    precision, recall = tp / max(1, len(p)), tp / max(1, len(t))
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "matched": tp, "predicted": len(p), "truth": len(t)}
