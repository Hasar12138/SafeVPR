"""Stage 1 (black-box) retrieval: diagonal-mean cosine pre-screen + DTW re-rank.

This stage maps a frame-level cosine similarity matrix to a per-database-window
sequence score, pre-screens the top candidates, applies spatial non-maximum
suppression, and re-ranks the surviving candidates by their DTW alignment cost.
SafeVPR consumes only the resulting scalar score / top-1 index; the retrieval
itself is treated as a replaceable black box.

The arithmetic here is lifted verbatim from the original ``process_query`` /
``dtw_with_path`` pipeline; only data loading was stripped out (it lives in the
``datasets`` package).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .dtw import dtw_with_path, extract_path_features

T_FRAMES = 10
K_NMS = 10
K_PRESCREEN = 50


def diagonal_mean_scores(S: np.ndarray, t: int = T_FRAMES) -> np.ndarray:
    """Per-database-window diagonal-mean cosine score.

    ``S`` is a frame-level similarity matrix. For a query window of length ``t``,
    the score of a database window starting at index ``d`` is the mean over the
    ``t`` diagonal frame pairs:

        score[d] = (1 / t) * sum_i  S[i, d + i]      (when S is t x N)

    More generally (``S`` of shape ``N_db x N_qr``, the form used by the
    sequence pipeline), with a query window starting at column ``q = 0`` the same
    diagonal mean is computed over rows. Both layouts reduce to averaging the
    diagonal of the aligned ``t x t`` block.

    Returns a 1-D array of length ``N_seq = N - t + 1`` (number of valid
    database window start positions).
    """
    S = np.asarray(S, dtype=np.float64)
    n_rows, n_cols = S.shape
    if n_rows == t:
        # query-window x database layout: row i = query frame i, col = db frame
        N = n_cols
        N_seq = N - t + 1
        diag = np.zeros(N_seq, dtype=np.float64)
        for i in range(t):
            diag += S[i, i:i + N_seq]
        diag /= t
        return diag
    # N_db x N_qr layout with query window anchored at column 0
    N_seq = n_rows - t + 1
    diag = np.zeros(N_seq, dtype=np.float64)
    for i in range(t):
        diag += S[i:i + N_seq, i]
    diag /= t
    return diag


def _spatial_nms(sorted_indices: Sequence[int], sorted_scores: Sequence[float],
                 k_nms: int, k_keep: int) -> Tuple[List[int], List[float]]:
    """Greedy NMS: accept a candidate iff separated from all kept by >= ``k_nms``."""
    kept_idx: List[int] = []
    kept_score: List[float] = []
    for idx, s in zip(sorted_indices, sorted_scores):
        if all(abs(idx - k) >= k_nms for k in kept_idx):
            kept_idx.append(int(idx))
            kept_score.append(float(s))
            if len(kept_idx) >= k_keep:
                break
    return kept_idx, kept_score


def dtw_rerank(S: np.ndarray, candidates: Sequence[int], t: int = T_FRAMES,
               q: int = 0) -> Tuple[int, float]:
    """Re-rank candidate database windows by DTW alignment and return the best.

    For each candidate start ``d`` in ``candidates`` the DTW cost over the aligned
    ``t x t`` block (cost = ``1 - cosine``) is computed and converted into the
    path ``best_score`` feature (``-best_cost / t``). The candidate with the
    highest ``best_score`` is returned as ``(best_index, best_score)``.

    ``S`` is the ``N_db x N_qr`` similarity matrix and ``q`` the query-window
    start column (default 0). Returns ``(-1, nan)`` if no candidate is valid.
    """
    S = np.asarray(S, dtype=np.float64)
    best_index = -1
    best_score = -np.inf
    for d in candidates:
        d = int(d)
        if d < 0:
            continue
        cm = 1.0 - S[d:d + t, q:q + t]
        cost, path = dtw_with_path(cm)
        feats = extract_path_features(cm, path, cost)
        if feats["best_score"] > best_score:
            best_score = feats["best_score"]
            best_index = d
    if best_index < 0:
        return -1, float("nan")
    return best_index, float(best_score)


def top_candidates(S: np.ndarray, t: int = T_FRAMES, k: int = 5,
                   k_nms: int = K_NMS, k_prescreen: int = K_PRESCREEN,
                   q: int = 0) -> Tuple[List[int], List[float]]:
    """Diagonal-mean pre-screen + spatial NMS, returning up to ``k`` candidate starts.

    Mirrors the candidate-selection stage of the original sequence pipeline:
    rank database windows by diagonal-mean cosine, take the top ``k_prescreen``,
    then apply spatial NMS with radius ``k_nms`` to keep ``k`` well-separated
    candidates.
    """
    diag = diagonal_mean_scores(S, t=t)
    n = len(diag)
    kk = min(k_prescreen, n)
    pre_idx = np.argpartition(diag, -kk)[-kk:]
    pre_idx = pre_idx[np.argsort(-diag[pre_idx])]
    pre_score = diag[pre_idx]
    return _spatial_nms(pre_idx, pre_score, k_nms, k)
