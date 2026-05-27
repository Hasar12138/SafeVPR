"""Oxford RobotCar dataset loading utilities.

- Load RTK positions (timestamp, northing, easting, yaw).
- Interpolate positions to image timestamps.
- Spatial subsample at a configurable distance.
- Ground-truth computation with strict / eval thresholds.
- Image preprocessing: raw Bayer -> demosaic(GRBG) -> undistort(LUT) -> resize -> 3-channel RGB.

All paths are supplied by the caller (e.g. loaded from ``configs/paths.yaml``);
nothing is hard-coded. Image timestamps stay int64 microseconds and are never
cast to float32.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

# Bayer pattern for the stereo cameras (from the RobotCar SDK).
BAYER_PATTERN = "gbrg"

# Traversal-id mapping (mirrors configs/oxford.yaml).
TRAVERSALS = {
    "database": "2014-12-09-13-21-02",
    "rain": "2015-10-29-12-18-17",
    "dusk": "2014-11-21-16-07-03",
    "overcast_rain": "2014-11-25-09-18-32",
    "overcast_3month": "2015-03-17-11-08-44",
    "night": "2014-12-16-18-44-24",
}


def load_rtk(root: Union[str, Path], traversal_id: str) -> dict:
    """Load RTK GPS data for a traversal under ``root``.

    Returns a dict with arrays: timestamp, northing, easting, yaw. Timestamps
    are int64 microseconds.
    """
    rtk_path = Path(root) / traversal_id / "rtk" / "rtk.csv"
    timestamps, northings, eastings, yaws = [], [], [], []
    with open(rtk_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(int(row["timestamp"]))
            northings.append(float(row["northing"]))
            eastings.append(float(row["easting"]))
            yaws.append(float(row["yaw"]))
    return {
        "timestamp": np.array(timestamps, dtype=np.int64),
        "northing": np.array(northings, dtype=np.float64),
        "easting": np.array(eastings, dtype=np.float64),
        "yaw": np.array(yaws, dtype=np.float64),
    }


def load_image_timestamps(root: Union[str, Path], traversal_id: str) -> np.ndarray:
    """Load stereo image timestamps (int64 microseconds)."""
    ts_path = Path(root) / traversal_id / "stereo.timestamps"
    timestamps = []
    with open(ts_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                timestamps.append(int(parts[0]))
    return np.array(timestamps, dtype=np.int64)


def interpolate_positions(image_ts: np.ndarray, rtk: dict) -> dict:
    """Interpolate RTK positions to image timestamps; keep only images in range."""
    rtk_ts = rtk["timestamp"]
    mask = (image_ts >= rtk_ts[0]) & (image_ts <= rtk_ts[-1])
    valid_ts = image_ts[mask]

    northing = np.interp(valid_ts, rtk_ts, rtk["northing"])
    easting = np.interp(valid_ts, rtk_ts, rtk["easting"])
    yaw = np.interp(valid_ts, rtk_ts, rtk["yaw"])

    return {
        "timestamp": valid_ts,
        "northing": northing,
        "easting": easting,
        "yaw": yaw,
        "valid_mask": mask,
    }


def spatial_subsample(positions: dict, min_dist_m: float = 2.0) -> np.ndarray:
    """Subsample positions to a minimum distance between consecutive frames."""
    n = len(positions["northing"])
    indices = [0]
    for i in range(1, n):
        last = indices[-1]
        dn = positions["northing"][i] - positions["northing"][last]
        de = positions["easting"][i] - positions["easting"][last]
        dist = np.sqrt(dn ** 2 + de ** 2)
        if dist >= min_dist_m:
            indices.append(i)
    return np.array(indices, dtype=np.int64)


def load_traversal(root: Union[str, Path], traversal_id: str,
                   subsample_m: float = 2.0):
    """Load a traversal: RTK interpolation + spatial subsampling."""
    root = Path(root)
    rtk = load_rtk(root, traversal_id)
    image_ts = load_image_timestamps(root, traversal_id)
    positions = interpolate_positions(image_ts, rtk)

    sub_idx = spatial_subsample(positions, subsample_m)

    img_dir = root / traversal_id / "stereo" / "left"
    img_paths = [img_dir / f"{positions['timestamp'][i]}.png" for i in sub_idx]

    return {
        "traversal_id": traversal_id,
        "timestamp": positions["timestamp"][sub_idx],
        "northing": positions["northing"][sub_idx],
        "easting": positions["easting"][sub_idx],
        "yaw": positions["yaw"][sub_idx],
        "image_paths": img_paths,
        "n_frames": len(sub_idx),
    }


def compute_distances(db_pos: dict, qr_pos: dict) -> np.ndarray:
    """Pairwise Euclidean distances (meters) between DB and query positions."""
    dn = db_pos["northing"][:, None] - qr_pos["northing"][None, :]
    de = db_pos["easting"][:, None] - qr_pos["easting"][None, :]
    return np.sqrt(dn ** 2 + de ** 2)


def compute_heading_diff(db_yaw: np.ndarray, qr_yaw: np.ndarray) -> np.ndarray:
    """Pairwise absolute heading difference (degrees), wrapped to [0, pi]."""
    diff = np.abs(db_yaw[:, None] - qr_yaw[None, :])
    diff = np.minimum(diff, 2 * np.pi - diff)
    return np.degrees(diff)


def compute_ground_truth(
    db_pos: dict, qr_pos: dict,
    pos_thresh_m: float = 10.0,
    neg_thresh_m: float = 25.0,
    heading_thresh_deg: float = 30.0,
) -> dict:
    """Ground-truth labels with strict training thresholds (1 / 0 / -1 ambiguous)."""
    dists = compute_distances(db_pos, qr_pos)
    heading_diff = compute_heading_diff(db_pos["yaw"], qr_pos["yaw"])

    best_idx = np.argmin(dists, axis=0)
    best_dist = dists[best_idx, np.arange(len(best_idx))]
    best_heading = heading_diff[best_idx, np.arange(len(best_idx))]

    labels = np.full(len(best_idx), -1, dtype=np.int32)
    pos_mask = (best_dist < pos_thresh_m) & (best_heading < heading_thresh_deg)
    labels[pos_mask] = 1
    neg_mask = best_dist > neg_thresh_m
    labels[neg_mask] = 0

    return {
        "best_match_idx": best_idx,
        "best_match_dist": best_dist,
        "best_heading_diff": best_heading,
        "labels": labels,
        "n_positive": int(pos_mask.sum()),
        "n_negative": int(neg_mask.sum()),
        "n_ambiguous": int((labels == -1).sum()),
    }


def compute_eval_labels(
    best_dist: np.ndarray, best_heading: np.ndarray,
    threshold_m: float, heading_thresh_deg: float = 30.0,
) -> np.ndarray:
    """Binary eval labels: 1 if dist < threshold AND heading ok."""
    return ((best_dist < threshold_m) & (best_heading < heading_thresh_deg)).astype(np.int32)


# ── Image preprocessing ──

_LUT_CACHE = None
_IMAGE_HW = (960, 1280)


def load_lut(lut_path: Union[str, Path]) -> tuple:
    """Load the undistortion LUT (cached) from ``lut_path``.

    The official SDK stores the LUT as a flat float64 buffer of length 2*H*W:
    the first H*W values are x-coords and the second H*W are y-coords.
    """
    global _LUT_CACHE
    if _LUT_CACHE is None:
        H, W = _IMAGE_HW
        flat = np.fromfile(str(lut_path), dtype=np.float64)
        lut = flat.reshape(2, -1).transpose()  # (H*W, 2): row k = (x_k, y_k)
        map_x = lut[:, 0].reshape(H, W).astype(np.float32)
        map_y = lut[:, 1].reshape(H, W).astype(np.float32)
        _LUT_CACHE = (map_x, map_y)
    return _LUT_CACHE


def preprocess_image(image_path: Union[str, Path], lut_path: Union[str, Path],
                     resize: tuple = (480, 640)) -> np.ndarray:
    """Full Oxford preprocessing: raw Bayer -> demosaic(GRBG) -> undistort(LUT) -> resize -> RGB.

    Returns an ``(H, W, 3)`` uint8 RGB array suitable for backbone input.
    """
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Demosaic (GBRG Bayer -> BGR)
    bgr = cv2.cvtColor(raw, cv2.COLOR_BayerGR2BGR)

    # Undistort using the LUT (per channel)
    map_x, map_y = load_lut(lut_path)
    b = cv2.remap(bgr[:, :, 0], map_x, map_y, cv2.INTER_LINEAR)
    g = cv2.remap(bgr[:, :, 1], map_x, map_y, cv2.INTER_LINEAR)
    r = cv2.remap(bgr[:, :, 2], map_x, map_y, cv2.INTER_LINEAR)
    bgr = np.stack([b, g, r], axis=-1)

    # Resize then return RGB
    bgr = cv2.resize(bgr, (resize[1], resize[0]))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
