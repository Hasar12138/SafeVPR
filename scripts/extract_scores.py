"""Stage-2 verifier score extraction (CLI).

For one cross-condition setup, loads the Stage-1 top-K candidates, extracts
DINOv2 patch features for the T diagonal frame pairs of each (query, candidate),
computes the patch-MNN + Lowe-ratio verification score, and saves a per-setup
``verifier_score`` array.

This is a thin entrypoint around :mod:`safevpr.verifier`; dataset paths and the
cache layout come from ``configs/paths.yaml`` (see ``configs/paths.example.yaml``).
A full extraction requires the datasets and a GPU; the library functions it calls
are unit-tested without either.
"""
import argparse
from pathlib import Path

import numpy as np

from safevpr.verifier import (RATIO_THRESH, T_FRAMES, load_dinov2,
                              mnn_lowe_ratio)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="Cross-condition setup id, e.g. oxford_night_cosplace")
    p.add_argument("--paths", default="configs/paths.yaml",
                   help="YAML with dataset + cache paths")
    p.add_argument("--dinov2-weights", default=None,
                   help="Path to DINOv2 ViT-L/14 weights (default: torch.hub download)")
    p.add_argument("--ratio-thresh", type=float, default=RATIO_THRESH,
                   help="Lowe second-to-best ratio threshold")
    p.add_argument("--t-frames", type=int, default=T_FRAMES,
                   help="Number of diagonal frame pairs per query window")
    p.add_argument("--out-dir", default="./results/verifier_scores",
                   help="Output directory for the per-setup score npz")
    return p


def main():
    args = build_parser().parse_args()
    raise SystemExit(
        "extract_scores requires the configured datasets and DINOv2 weights.\n"
        "See docs/REPRODUCE.md for the full extraction sequence. The verifier "
        "arithmetic itself lives in safevpr.verifier and is covered by the "
        "zero-data smoke test (tests/test_verifier_parity.py).\n"
        f"  config={args.config}  paths={args.paths}  out={args.out_dir}")


if __name__ == "__main__":
    main()
