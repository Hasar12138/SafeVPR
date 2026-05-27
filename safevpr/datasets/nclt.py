"""NCLT dataset loading utilities.

NCLT is a campus multi-session dataset (perceptual-aliasing stress test). Each
session directory contains ``lb3/Cam5_converted/*.png`` (filenames are
microsecond timestamps) and a ``groundtruth.csv`` of 6-DoF poses.

This module lifts the previously inline image-listing + ground-truth-parsing
logic into clean, path-from-config functions. The Cam5_converted images are
already undistorted. Timestamps stay int64 microseconds and are never cast to
float32.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np

IMAGE_SUBDIR = "lb3/Cam5_converted"
GT_FILENAME = "groundtruth.csv"
SUBSAMPLE_M = 2.0


def load_session_data(root: Union[str, Path], session_id: str,
                      subsample_m: float = SUBSAMPLE_M) -> Tuple[list, np.ndarray, np.ndarray]:
    """Load one NCLT session: images, ground-truth-interpolated XY, timestamps.

    Steps: list images (sorted by timestamp filename), parse ground-truth poses,
    drop NaN rows, interpolate (x, y) to image timestamps, then spatially
    subsample at ``subsample_m`` meters.

    Returns ``(image_paths, positions_xy, timestamps)`` where ``positions_xy`` is
    an ``(N, 2)`` array and ``timestamps`` is int64 microseconds.
    """
    sess_dir = Path(root) / session_id
    img_dir = sess_dir / IMAGE_SUBDIR
    gt_path = sess_dir / GT_FILENAME

    # Ground truth: columns are (timestamp, x, y, z, roll, pitch, yaw).
    gt_data = np.genfromtxt(gt_path, delimiter=",", skip_header=1)
    valid = ~np.isnan(gt_data[:, 1])
    gt = gt_data[valid]
    gt_ts = gt[:, 0].astype(np.int64)
    gt_x = gt[:, 1]
    gt_y = gt[:, 2]

    img_paths = sorted(img_dir.glob("*.png"))
    img_ts = np.array([int(p.stem) for p in img_paths], dtype=np.int64)

    # Interpolate ground-truth XY to image timestamps.
    x_interp = np.interp(img_ts.astype(np.float64), gt_ts.astype(np.float64), gt_x)
    y_interp = np.interp(img_ts.astype(np.float64), gt_ts.astype(np.float64), gt_y)

    # Spatial subsample at subsample_m.
    keep = [0]
    for i in range(1, len(img_paths)):
        dx = x_interp[i] - x_interp[keep[-1]]
        dy = y_interp[i] - y_interp[keep[-1]]
        if np.sqrt(dx * dx + dy * dy) >= subsample_m:
            keep.append(i)
    keep = np.array(keep)

    paths_sub = [img_paths[i] for i in keep]
    pos_sub = np.column_stack([x_interp[keep], y_interp[keep]])
    ts_sub = img_ts[keep]
    return paths_sub, pos_sub, ts_sub
