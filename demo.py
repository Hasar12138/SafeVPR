"""Zero-data demo of the SafeVPR accept/abstain decision.

Synthesises verifier scores + ground-truth correctness (no images, no GPU, no
datasets), then runs Stage 3 of the pipeline: fit the Mondrian conformal
Learn-Then-Test thresholds on a calibration split and decide accept/abstain on a
test split. Prints the controlled false-discovery rate (FDR), the true-positive
rate (TPR), and a few example per-query decisions.

Run:
    python demo.py
"""
import numpy as np

from safevpr.conformal import accept, fit_mondrian_ltt
from safevpr.metrics import coverage, fdr, tpr


def synthesise(n=2000, seed=0):
    """High verifier score -> usually a correct match; low score -> usually wrong."""
    rng = np.random.default_rng(seed)
    score = np.concatenate([rng.uniform(0.55, 1.0, n // 2),
                            rng.uniform(0.0, 0.5, n // 2)])
    correct = np.concatenate([rng.random(n // 2) < 0.95,
                              rng.random(n // 2) < 0.08]).astype(int)
    order = rng.permutation(n)
    return score[order], correct[order]


def main():
    score, correct = synthesise()
    cal_s, cal_c = score[:1000], correct[:1000]
    test_s, test_c = score[1000:], correct[1000:]

    print("SafeVPR demo  --  synthetic verifier scores, no data/GPU")
    print(f"calibration: {len(cal_s)} queries   test: {len(test_s)} queries\n")
    print(f"{'alpha':>6} | {'accepted':>8} | {'coverage':>8} | {'FDR':>6} | "
          f"{'<= alpha?':>9} | {'TPR':>6}")
    print("-" * 60)
    for alpha in (0.05, 0.10):
        thr = fit_mondrian_ltt(cal_s, cal_c, alpha)
        acc = accept(test_s, thr)
        f = fdr(test_c, acc)
        print(f"{alpha:>6.2f} | {int(acc.sum()):>8} | {coverage(acc):>8.3f} | "
              f"{f:>6.3f} | {'yes' if f <= alpha + 1e-9 else 'NO':>9} | "
              f"{tpr(test_c, acc):>6.3f}")

    # A few example per-query decisions at alpha = 0.10
    thr = fit_mondrian_ltt(cal_s, cal_c, 0.10)
    acc = accept(test_s, thr)
    print("\nexample decisions (alpha = 0.10):")
    print(f"{'score s':>8} | {'decision':>8} | {'ground truth':>12} | outcome")
    print("-" * 56)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(test_s), 6, replace=False):
        decision = "ACCEPT" if acc[i] else "ABSTAIN"
        truth = "correct" if test_c[i] else "wrong"
        if acc[i] and test_c[i]:
            outcome = "true positive"
        elif acc[i] and not test_c[i]:
            outcome = "false discovery (bounded by alpha)"
        elif not acc[i] and test_c[i]:
            outcome = "lost recall"
        else:
            outcome = "safely rejected"
        print(f"{test_s[i]:>8.3f} | {decision:>8} | {truth:>12} | {outcome}")


if __name__ == "__main__":
    main()
