"""Demo of SafeVPR's cross-condition accept/abstain decision on real scores.

Replicates the paper's cross-condition protocol on a small bundled sample of
REAL patch-MNN verifier scores: the eight CosPlace-backbone deployments
(``examples/cosplace_scores/*.npz``; Oxford RobotCar dusk/night/overcast/rain,
NCLT three sessions, St Lucia). For each held-out deployment the Mondrian
conformal thresholds are calibrated on the *other* deployments' calibration
slices and evaluated on the held-out deployment — so calibration and test never
share a condition. No images, GPU, or full datasets are needed.

It prints the leave-one-out validity (how many deployments keep FDR <= alpha),
then a detailed view of one hard held-out deployment (Oxford night) with example
per-query decisions.

Run:
    python demo.py
"""
from pathlib import Path

import numpy as np

from safevpr.conformal import accept, fit_mondrian_ltt
from safevpr.evaluation import (collect_calibration, collect_test,
                                evaluate_setup)
from safevpr.metrics import fdr, tpr

SCORES = Path(__file__).parent / "examples" / "cosplace_scores"
FOCUS = "oxford_night_cosplace"
ALPHAS = (0.05, 0.10)


def main():
    setups = sorted(p.stem for p in SCORES.glob("*.npz"))

    print("SafeVPR demo  --  real cross-condition protocol (CosPlace backbone)")
    print(f"{len(setups)} deployments; for each, calibrate on the other "
          f"{len(setups) - 1}, evaluate on the held-out one.\n")

    print("Leave-one-out validity (FDR controlled per held-out deployment):")
    print(f"{'alpha':>6} | {'valid':>7} | {'mean FDR':>8} | {'mean TPR':>8}")
    print("-" * 40)
    for alpha in ALPHAS:
        res = []
        for held_out in setups:
            train = [s for s in setups if s != held_out]
            cal_s, cal_c = collect_calibration(SCORES, train, "verifier")
            te_s, te_c = collect_test(SCORES, held_out, "verifier")
            res.append(evaluate_setup(cal_s, cal_c, te_s, te_c, alpha, "mondrian"))
        n_valid = sum(r["valid"] for r in res)
        mean_fdr = np.mean([r["FDR"] for r in res if r["n_accept"] > 0])
        mean_tpr = np.mean([r["TPR"] for r in res])
        print(f"{alpha:>6.2f} | {n_valid:>4}/{len(setups):<2} | "
              f"{mean_fdr:>8.3f} | {mean_tpr:>8.3f}")

    # Detailed view of one hard held-out deployment
    train = [s for s in setups if s != FOCUS]
    cal_s, cal_c = collect_calibration(SCORES, train, "verifier")
    te_s, te_c = collect_test(SCORES, FOCUS, "verifier")
    print(f"\nDetailed view  --  held out: Oxford night (calibrated on the "
          f"other {len(train)} deployments)")
    print(f"{'alpha':>6} | {'accepted':>8} | {'coverage':>8} | {'FDR':>6} | "
          f"{'<= alpha?':>9} | {'TPR':>6}")
    print("-" * 60)
    for alpha in ALPHAS:
        r = evaluate_setup(cal_s, cal_c, te_s, te_c, alpha, "mondrian")
        cov = r["n_accept"] / r["n_total"]
        print(f"{alpha:>6.2f} | {r['n_accept']:>8} | {cov:>8.3f} | "
              f"{r['FDR']:>6.3f} | {'yes' if r['valid'] else 'NO':>9} | "
              f"{r['TPR']:>6.3f}")

    thr = fit_mondrian_ltt(cal_s, cal_c, 0.10)
    acc = accept(te_s, thr)
    print("\nexample decisions (held-out Oxford night, alpha = 0.10):")
    print(f"{'score s':>8} | {'decision':>8} | {'ground truth':>12} | outcome")
    print("-" * 56)
    # one representative query from each outcome type that occurs
    groups = [
        (acc & (te_c == 1), "ACCEPT", "correct", "true positive"),
        (acc & (te_c == 0), "ACCEPT", "wrong", "false discovery (bounded by alpha)"),
        (~acc & (te_c == 0), "ABSTAIN", "wrong", "safely rejected"),
        (~acc & (te_c == 1), "ABSTAIN", "correct", "lost recall"),
    ]
    rng = np.random.default_rng(0)
    for mask, decision, truth, outcome in groups:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        for i in rng.choice(idx, min(2, len(idx)), replace=False):
            print(f"{te_s[i]:>8.3f} | {decision:>8} | {truth:>12} | {outcome}")


if __name__ == "__main__":
    main()
