from tests.fixture_synthetic import make_scores_labels
from safevpr.conformal import fit_mondrian_ltt, accept
from safevpr.metrics import fdr, tpr


def test_end_to_end_calibrate_evaluate():
    s, y = make_scores_labels()
    mid = len(s) // 2
    cal_s, cal_y, te_s, te_y = s[:mid], y[:mid], s[mid:], y[mid:]
    th = fit_mondrian_ltt(cal_s, cal_y, alpha=0.10, delta=0.05, n_bins=5, grid_m=5)
    acc = accept(te_s, th)
    assert fdr(te_y, acc) <= 0.10 + 0.05   # empirical slack on a tiny sample
    assert tpr(te_y, acc) > 0.0
