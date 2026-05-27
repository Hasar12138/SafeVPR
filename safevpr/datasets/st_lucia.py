"""St Lucia dataset loading utilities.

St Lucia is a morning-vs-noon single-route pair. Image filenames encode UTM
coordinates and a panoid that recovers the original video-frame order:

    @easting@northing@zone@J@lat@lon@panoid@@@@@@@@.jpg

This module lifts the previously inline filename-parsing + traversal-listing
logic into clean, path-from-config functions. Sorting by the panoid field (the
7th ``@``-separated token) yields temporal frame order.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


def parse_filename(fname: str) -> Tuple[float, float, str]:
    """Parse an St Lucia filename into ``(easting, northing, panoid)``."""
    parts = fname.split("@")
    return float(parts[1]), float(parts[2]), parts[7]


def list_traversal(directory: Union[str, Path]) -> List[Tuple[Path, float, float, str]]:
    """List a traversal as ``(path, easting, northing, panoid)`` sorted by panoid.

    The panoid encodes the original video-frame index, giving temporal order.
    """
    directory = Path(directory)
    out: List[Tuple[Path, float, float, str]] = []
    for p in directory.glob("*.jpg"):
        e, n, pid = parse_filename(p.name)
        out.append((p, e, n, pid))
    out.sort(key=lambda r: r[3])
    return out


def load_traversal(directory: Union[str, Path]) -> Tuple[List[Path], np.ndarray]:
    """Load a traversal: return ``(image_paths, positions_xy)`` in temporal order.

    ``positions_xy`` is an ``(N, 2)`` array of ``(easting, northing)`` in meters.
    """
    records = list_traversal(directory)
    paths = [r[0] for r in records]
    pos = np.array([(r[1], r[2]) for r in records], dtype=np.float64)
    if len(pos) == 0:
        pos = pos.reshape(0, 2)
    return paths, pos
