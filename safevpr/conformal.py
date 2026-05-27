"""Stage 3 calibrator: Mondrian conformal Learn-Then-Test (LTT).

The verifier score ``s`` is calibrated into a fixed-FDR accept/reject decision.
Queries are partitioned into ``N_BINS`` quantile (quintile) bins of the score,
and a per-bin threshold ``tau_b`` is fitted by an LTT procedure with a
Bonferroni split of the error budget ``delta`` over the bins. The per-bin
threshold is the smallest score whose Clopper-Pearson FDR upper bound is at most
``alpha``.

All functions are pure (numpy in, numpy out) -- no I/O, no file paths.

Numerical core (must match the paper):
  N_BINS = 5, DELTA = 0.05, GRID_M = 5, ALPHAS = (0.05, 0.10), per-bin tau.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as sp_beta

# ── numerical core ──
N_BINS = 5            # Mondrian quintile binning
DELTA = 0.05
GRID_M = 5            # LTT Bonferroni grid size
ALPHAS = (0.05, 0.10)


def cp_upper(w: int, n: int, delta: float) -> float:
    """Clopper-Pearson upper confidence bound on the error rate.

    ``w`` errors out of ``n`` accepted, at confidence ``1 - delta``.
    """
    if n == 0 or w >= n:
        return 1.0
    return float(sp_beta.ppf(1.0 - delta, w + 1, n - w))


def quintile_bins(scores: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Quantile bin edges of ``scores`` with extremes extended to +/- inf."""
    edges = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def fit_mondrian_ltt(cal_scores: np.ndarray, cal_labels: np.ndarray, alpha: float,
                     delta: float = DELTA, n_bins: int = N_BINS,
                     grid_m: int = GRID_M) -> dict:
    """Fit per-bin LTT thresholds (Mondrian conformal calibration).

    Queries are binned by quantile of ``cal_scores``. Within each bin the
    smallest threshold whose FDR upper bound (Clopper-Pearson, Bonferroni-split
    ``delta / n_bins``) is at most ``alpha`` is selected as that bin's ``tau``.

    ``grid_m`` is accepted for interface compatibility; per-bin search scans all
    candidate thresholds (matching the original Mondrian fit).

    Returns a dict with ``"edges"`` (``n_bins + 1`` bin edges) and ``"taus"``
    (``n_bins`` per-bin thresholds; ``inf`` for empty / undecidable bins).
    """
    bin_var = cal_scores
    bin_edges = quintile_bins(bin_var, n_bins)
    taus = np.full(n_bins, np.inf)
    for b in range(n_bins):
        in_bin = (bin_var >= bin_edges[b]) & (bin_var < bin_edges[b + 1])
        if in_bin.sum() < 5:
            continue  # skip nearly-empty bins
        s_b = cal_scores[in_bin]
        c_b = cal_labels[in_bin]
        delta_per = delta / max(n_bins, 1)
        n = len(s_b)
        order = np.argsort(-s_b)
        ss = s_b[order]
        ys = c_b[order]
        cum = np.cumsum(ys)
        k = np.arange(1, n + 1)
        n_w = k - cum
        best = (0, np.inf)
        for kk in range(1, n + 1):
            ucb = cp_upper(int(n_w[kk - 1]), int(kk), delta_per)
            if ucb <= alpha and kk > best[0]:
                best = (int(kk), float(ss[kk - 1]))
        taus[b] = best[1]
    return {"edges": bin_edges, "taus": taus, "n_bins": int(n_bins)}


def calibrate_ltt(scores: np.ndarray, correct: np.ndarray, alpha: float,
                  delta: float = DELTA, grid_m: int = GRID_M):
    """Global (non-Mondrian) LTT threshold with a Bonferroni grid of size ``grid_m``.

    Returns ``(tau, k)`` where ``tau`` is the chosen score threshold (``inf`` if
    no threshold controls FDR) and ``k`` is the number of accepted cal points.
    """
    n = len(scores)
    if n == 0:
        return float("inf"), 0
    order = np.argsort(-scores)
    ss = scores[order]
    ys = correct[order]
    cum = np.cumsum(ys)
    k = np.arange(1, n + 1)
    n_w = k - cum
    delta_per = delta / max(grid_m, 1)
    grid = np.unique(np.linspace(1, n, grid_m).astype(int))
    best = (0, float("inf"))
    for kk in grid:
        idx = kk - 1
        ucb = cp_upper(int(n_w[idx]), int(kk), delta_per)
        if ucb <= alpha and kk > best[0]:
            best = (int(kk), float(ss[idx]))
    return best[1], best[0]


def calibrate_cp_simple(scores: np.ndarray, correct: np.ndarray, alpha: float,
                        delta: float = DELTA):
    """Simple single-bound conformal threshold (full scan, no Bonferroni grid)."""
    n = len(scores)
    if n == 0:
        return float("inf"), 0
    order = np.argsort(-scores)
    ss = scores[order]
    ys = correct[order]
    cum = np.cumsum(ys)
    k = np.arange(1, n + 1)
    n_w = k - cum
    best = (0, float("inf"))
    for kk in range(1, n + 1):
        ucb = cp_upper(int(n_w[kk - 1]), int(kk), delta)
        if ucb <= alpha and kk > best[0]:
            best = (int(kk), float(ss[kk - 1]))
    return best[1], best[0]


def accept_global(test_scores: np.ndarray, tau: float) -> np.ndarray:
    """Accept mask for a single global threshold ``tau``."""
    return np.asarray(test_scores) >= tau


def accept(test_scores: np.ndarray, thresholds: dict) -> np.ndarray:
    """Apply per-bin Mondrian thresholds to ``test_scores``; return a bool mask.

    Each test query is assigned to its bin via ``np.digitize`` on the calibrated
    edges, then accepted iff its score is at least that bin's ``tau``.
    """
    edges = np.asarray(thresholds["edges"])
    taus = np.asarray(thresholds["taus"])
    s = np.asarray(test_scores)
    bins = np.digitize(s, edges) - 1
    bins = np.clip(bins, 0, len(taus) - 1)
    return np.array([s[i] >= taus[bins[i]] for i in range(len(s))], dtype=bool)
