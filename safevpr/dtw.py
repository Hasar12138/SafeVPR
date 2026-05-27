"""DTW alignment and path-confidence features for the Stage-1 sequence re-rank.

Standard dynamic time warping with backtracking over a (lower-is-better) cost
matrix, plus the path-derived confidence features used to re-rank candidate
database windows. Arithmetic is preserved exactly from the original matcher.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def dtw_with_path(cost_matrix: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    """Standard DTW with backtracking to recover the optimal path.

    Args:
        cost_matrix: ``(T1, T2)`` pairwise distances (lower = better match).

    Returns:
        ``(total_cost, path)`` where ``path`` is a list of ``(i, j)`` tuples.
    """
    T1, T2 = cost_matrix.shape
    R = np.full((T1, T2), np.inf, dtype=np.float64)
    R[0, 0] = cost_matrix[0, 0]

    for i in range(1, T1):
        R[i, 0] = R[i - 1, 0] + cost_matrix[i, 0]
    for j in range(1, T2):
        R[0, j] = R[0, j - 1] + cost_matrix[0, j]
    for i in range(1, T1):
        for j in range(1, T2):
            R[i, j] = cost_matrix[i, j] + min(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1])

    path = []
    i, j = T1 - 1, T2 - 1
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [R[i - 1, j - 1], R[i - 1, j], R[i, j - 1]]
            idx = int(np.argmin(candidates))
            if idx == 0:
                i -= 1
                j -= 1
            elif idx == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    return R[T1 - 1, T2 - 1], path


def dtw_with_exclusion(cost_matrix: np.ndarray, path: List[Tuple[int, int]],
                       margin: int = 2) -> float:
    """Approximate the second-best alignment cost by penalising cells near ``path``."""
    T1, T2 = cost_matrix.shape
    penalty = cost_matrix.max() * 2.0
    cost_mod = cost_matrix.copy()
    for pi, pj in path:
        for di in range(-margin, margin + 1):
            for dj in range(-margin, margin + 1):
                ni, nj = pi + di, pj + dj
                if 0 <= ni < T1 and 0 <= nj < T2:
                    cost_mod[ni, nj] = penalty

    R = np.full((T1, T2), np.inf, dtype=np.float64)
    R[0, 0] = cost_mod[0, 0]
    for i in range(1, T1):
        R[i, 0] = R[i - 1, 0] + cost_mod[i, 0]
    for j in range(1, T2):
        R[0, j] = R[0, j - 1] + cost_mod[0, j]
    for i in range(1, T1):
        for j in range(1, T2):
            R[i, j] = cost_mod[i, j] + min(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1])
    return R[T1 - 1, T2 - 1]


def extract_path_features(cost_matrix: np.ndarray, path: List[Tuple[int, int]],
                          best_cost: float) -> dict:
    """Extract 6 confidence features from the DTW cost matrix and optimal path.

    Returns a dict with keys: ``best_score``, ``cost_gap``, ``path_smoothness``,
    ``cost_variance``, ``alignment_ratio``, ``warping_degree``.
    """
    T = cost_matrix.shape[0]

    # 1. best_score (inverted, normalized by T)
    best_score = -best_cost / T

    # 2. cost_gap (uniqueness of best match)
    second_cost = dtw_with_exclusion(cost_matrix, path, margin=2)
    cost_gap = (second_cost - best_cost) / (best_cost + 1e-8)

    # 3. path_smoothness (fraction of diagonal steps)
    if len(path) > 1:
        steps = [(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
                 for k in range(len(path) - 1)]
        diagonal_steps = sum(1 for dx, dy in steps if dx == 1 and dy == 1)
        path_smoothness = diagonal_steps / len(steps)
    else:
        path_smoothness = 1.0

    # 4. cost_variance (per-step cost std)
    step_costs = np.array([cost_matrix[i, j] for i, j in path])
    cost_variance = float(np.std(step_costs))

    # 5. alignment_ratio (frames used / T)
    frames_used = len(set(i for i, j in path))
    alignment_ratio = frames_used / T

    # 6. warping_degree (mean deviation from diagonal)
    deviations = np.array([abs(i - j) for i, j in path])
    warping_degree = float(np.mean(deviations))

    return {
        "best_score": best_score,
        "cost_gap": cost_gap,
        "path_smoothness": path_smoothness,
        "cost_variance": cost_variance,
        "alignment_ratio": alignment_ratio,
        "warping_degree": warping_degree,
    }


FEATURE_NAMES = [
    "best_score", "cost_gap", "path_smoothness",
    "cost_variance", "alignment_ratio", "warping_degree",
]
