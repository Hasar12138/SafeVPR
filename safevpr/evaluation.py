"""Cross-condition evaluation helpers shared by the CLI entrypoints.

These functions load precomputed per-setup score arrays from a cache directory
and run the calibrate-then-evaluate loop using the library's conformal and
metric primitives. Cache layout and dataset paths are supplied by the caller
(via ``configs/paths.yaml``); nothing is hard-coded.

Cache layout expected (one ``.npz`` per setup under ``score_cache``):
    <setup>.npz with keys:
        verifier_score : (Nq, K) float   patch-MNN + Lowe score per candidate
        cosine         : (Nq,)   float    Stage-1 top-1 diagonal cosine
        correct        : (Nq,)   int      0/1 correctness of the top-1 candidate
        split          : (Nq,)   int      1 = calibration slice, else test/other
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .conformal import (ALPHAS, DELTA, GRID_M, N_BINS, accept, accept_global,
                        calibrate_ltt, fit_mondrian_ltt)
from .metrics import coverage, fdr, tpr


def load_setup_scores(score_cache: Path, setup: str) -> Dict[str, np.ndarray]:
    """Load the score arrays for one cross-condition setup from ``score_cache``."""
    d = np.load(Path(score_cache) / f"{setup}.npz", allow_pickle=True)
    out = {}
    if "verifier_score" in d:
        vs = d["verifier_score"]
        out["verifier"] = vs[:, 0] if vs.ndim == 2 else vs
    if "cosine" in d:
        out["cosine"] = d["cosine"]
    out["correct"] = d["correct"].astype(int)
    out["split"] = d["split"].astype(int)
    return out


def collect_calibration(score_cache: Path, setups: List[str], score: str = "verifier"):
    """Concatenate the calibration slice (split == 1) across ``setups``."""
    s_all, c_all = [], []
    for setup in setups:
        d = load_setup_scores(score_cache, setup)
        if score not in d:
            continue
        sel = np.where(d["split"] == 1)[0]
        s_all.append(d[score][sel])
        c_all.append(d["correct"][sel])
    s = np.concatenate(s_all)
    c = np.concatenate(c_all)
    keep = ~np.isnan(s)
    return s[keep], c[keep]


def collect_test(score_cache: Path, setup: str, score: str = "verifier"):
    """Load the full held-out setup data."""
    d = load_setup_scores(score_cache, setup)
    s = d[score]
    c = d["correct"]
    keep = ~np.isnan(s)
    return s[keep], c[keep]


def evaluate_setup(cal_scores, cal_labels, test_scores, test_labels, alpha: float,
                   calibrator: str = "mondrian") -> dict:
    """Calibrate on cal, evaluate on test; return FDR / TPR / coverage / validity."""
    if calibrator == "mondrian":
        th = fit_mondrian_ltt(cal_scores, cal_labels, alpha, delta=DELTA,
                              n_bins=N_BINS, grid_m=GRID_M)
        acc = accept(test_scores, th)
    elif calibrator == "ltt":
        tau, _ = calibrate_ltt(cal_scores, cal_labels, alpha, delta=DELTA, grid_m=GRID_M)
        acc = accept_global(test_scores, tau)
    else:
        raise ValueError(f"unknown calibrator: {calibrator}")
    n_accept = int(acc.sum())
    return {
        "alpha": float(alpha),
        "calibrator": calibrator,
        "coverage": coverage(acc),
        "FDR": fdr(test_labels, acc),
        "TPR": tpr(test_labels, acc),
        "n_accept": n_accept,
        "n_total": int(len(test_scores)),
        "valid": bool(n_accept == 0 or fdr(test_labels, acc) <= alpha),
    }
