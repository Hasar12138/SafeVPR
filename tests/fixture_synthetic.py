"""Synthetic verifier scores + labels for zero-data, zero-GPU smoke testing."""
import numpy as np


def make_scores_labels(n=2000, seed=0):
    """High-score -> mostly-correct, low-score -> mostly-wrong synthetic scores.

    The two score regimes are interleaved (shuffled) so that any contiguous
    cal/test split contains both classes -- otherwise a head split would be all
    positives and the conformal calibrator would have no negatives to bound.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    s = np.concatenate([rng.uniform(0.6, 1.0, half), rng.uniform(0.0, 0.5, half)])
    # High-score regime is clean enough (and the sample large enough) that the
    # per-bin conformal LTT can certify FDR <= alpha on the top quantile bin.
    y = np.concatenate([(rng.random(half) < 0.97), (rng.random(half) < 0.03)]).astype(int)
    perm = rng.permutation(len(s))
    return s[perm], y[perm]
