"""Cross-condition FDR-validity evaluation (CLI).

For each held-out setup, calibrates the conformal threshold(s) on the union of
the other same-backbone setups' calibration slices and evaluates FDR / TPR /
coverage / validity on the held-out test data. Reports per-setup results and an
aggregate pass count, for both the global LTT and the Mondrian calibrator, at
each alpha.

Thin entrypoint around :mod:`safevpr.evaluation`. Reads precomputed per-setup
score npz files from a cache directory (see ``safevpr.evaluation`` for the
expected layout). No internal codenames; the verifier score is referred to as
``s``.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from safevpr.conformal import ALPHAS
from safevpr.evaluation import (collect_calibration, collect_test,
                                evaluate_setup, load_setup_scores)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--score-cache", default="./results/verifier_scores",
                   help="Directory of per-setup score npz files")
    p.add_argument("--score", default="verifier", choices=["verifier", "cosine"],
                   help="Which score to calibrate")
    p.add_argument("--setups", nargs="+", default=None,
                   help="Explicit setup ids; default: every npz in --score-cache")
    p.add_argument("--backbone-of", default="suffix",
                   help="How to group setups by backbone (suffix = last _token)")
    p.add_argument("--out", default="./results/cross_condition/all_results.json",
                   help="Output JSON path")
    return p


def _discover_setups(score_cache: Path):
    return sorted(p.stem for p in Path(score_cache).glob("*.npz"))


def _backbone(setup: str) -> str:
    return setup.rsplit("_", 1)[1]


def main():
    args = build_parser().parse_args()
    cache = Path(args.score_cache)
    setups = args.setups or _discover_setups(cache)
    all_res = []
    for held_out in setups:
        bb = _backbone(held_out)
        train = [s for s in setups if s != held_out and _backbone(s) == bb]
        if not train:
            continue
        cal_s, cal_c = collect_calibration(cache, train, args.score)
        te_s, te_c = collect_test(cache, held_out, args.score)
        res = {"held_out": held_out, "backbone": bb,
               "Ncal": int(len(cal_s)), "Ntest": int(len(te_s)), "alphas": {}}
        for alpha in ALPHAS:
            res["alphas"][f"alpha_{alpha}"] = {
                "ltt": evaluate_setup(cal_s, cal_c, te_s, te_c, alpha, "ltt"),
                "mondrian": evaluate_setup(cal_s, cal_c, te_s, te_c, alpha, "mondrian"),
            }
        all_res.append(res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_res, indent=2))

    for alpha in ALPHAS:
        for calib in ("ltt", "mondrian"):
            n_pass = sum(1 for r in all_res
                         if r["alphas"][f"alpha_{alpha}"][calib]["valid"])
            print(f"alpha={alpha}  {calib:<8s}  pass={n_pass}/{len(all_res)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
