"""Evaluation metrics: FDR / TPR / coverage / AUROC.

These match the formulas used in the cross-condition evaluation. ``labels`` are
binary 0/1 correctness; ``accept_mask`` is a boolean accept/reject decision.
By convention an empty accept set has FDR 0.0 (vacuously valid) and TPR 0.0.
"""
from __future__ import annotations

import numpy as np


def fdr(labels: np.ndarray, accept_mask: np.ndarray) -> float:
    """False-discovery rate among accepted: 1 - (accepted correct / accepted)."""
    labels = np.asarray(labels)
    accept_mask = np.asarray(accept_mask, dtype=bool)
    n = int(accept_mask.sum())
    if n == 0:
        return 0.0
    return float(1.0 - int(labels[accept_mask].sum()) / n)


def tpr(labels: np.ndarray, accept_mask: np.ndarray) -> float:
    """True-positive rate: accepted positives / all positives."""
    labels = np.asarray(labels)
    accept_mask = np.asarray(accept_mask, dtype=bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0
    return float(int(labels[accept_mask].sum()) / n_pos)


def coverage(accept_mask: np.ndarray) -> float:
    """Fraction of queries accepted."""
    accept_mask = np.asarray(accept_mask, dtype=bool)
    if len(accept_mask) == 0:
        return 0.0
    return float(int(accept_mask.sum()) / len(accept_mask))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (rank-based).

    Returns ``nan`` if either class is absent. Ties contribute 0.5.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    # average ranks for ties (ranks are 1-indexed)
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # mean of ranks (i+1 .. j), 1-indexed
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = ranks[labels == 1].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))
