import numpy as np

from safevpr.conformal import fit_mondrian_ltt, accept


def test_mondrian_accepts_high_scores_only():
    rng = np.random.default_rng(0)
    # 400 cal: high score => mostly correct, low score => mostly wrong
    s = np.concatenate([rng.uniform(0.7, 1.0, 200), rng.uniform(0.0, 0.4, 200)])
    y = np.concatenate([rng.random(200) < 0.95, rng.random(200) < 0.05]).astype(int)
    th = fit_mondrian_ltt(s, y, alpha=0.10, delta=0.05, n_bins=5, grid_m=5)
    acc = accept(s, th)
    # accepted set FDR must respect alpha
    if acc.sum() > 0:
        fdr = 1 - y[acc].mean()
        assert fdr <= 0.10 + 1e-9
