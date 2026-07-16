"""Reference polygons and partitions used by the Section 7/9 demos and tests."""

from __future__ import annotations

import numpy as np


SECTION7_POLYGONS = {
    "heart_like": np.array([
        [-2.40, -0.20],
        [-1.90, -1.05],
        [-1.00, -1.70],
        [0.10, -1.95],
        [1.30, -1.55],
        [2.15, -0.75],
        [2.45, 0.10],
        [1.75, 0.82],
        [0.85, 0.98],
        [0.10, 0.58],
        [-0.70, 0.96],
        [-1.75, 0.84],
        [-2.45, 0.25],
    ], dtype=float),
    "narrow_neck": np.array([
        [-2.30, -1.40],
        [-1.35, -2.25],
        [-0.10, -2.55],
        [1.25, -2.20],
        [2.18, -1.35],
        [2.42, -0.18],
        [1.45, 0.20],
        [0.62, 0.82],
        [0.42, 1.72],
        [-0.42, 1.78],
        [-0.64, 0.90],
        [-1.58, 0.22],
        [-2.34, -0.32],
    ], dtype=float),
}

SECTION7_TEST_POINTS = {
    "heart_like": {
        "inside": [(-1.0, -0.8), (0.8, -1.0), (0.0, 0.0)],
        "outside": [(0.0, 1.15), (2.7, 0.3), (-2.7, 0.3)],
    },
    "narrow_neck": {
        "inside": [(-0.8, -1.5), (0.9, -1.4), (0.0, 0.2)],
        "outside": [(0.0, 2.1), (2.6, -0.4), (-2.6, -0.4)],
    },
}


PARTITION_OUTER = np.array([
    [0.00, 0.00],
    [4.10, 0.20],
    [4.80, 1.85],
    [4.05, 3.55],
    [2.15, 4.25],
    [0.15, 2.85],
], dtype=float)

PARTITION_INNER = np.array([
    [1.00, 1.00],
    [2.55, 1.10],
    [3.35, 2.00],
    [2.75, 3.05],
    [1.70, 3.30],
    [0.92, 2.18],
], dtype=float)

PARTITION_CELLS = [
    PARTITION_INNER.copy(),
    np.array([PARTITION_OUTER[0], PARTITION_OUTER[1], PARTITION_INNER[1], PARTITION_INNER[0]], dtype=float),
    np.array([PARTITION_OUTER[1], PARTITION_OUTER[2], PARTITION_INNER[2], PARTITION_INNER[1]], dtype=float),
    np.array([PARTITION_OUTER[2], PARTITION_OUTER[3], PARTITION_INNER[3], PARTITION_INNER[2]], dtype=float),
    np.array([PARTITION_OUTER[3], PARTITION_OUTER[4], PARTITION_INNER[4], PARTITION_INNER[3]], dtype=float),
    np.array([PARTITION_OUTER[4], PARTITION_OUTER[5], PARTITION_INNER[5], PARTITION_INNER[4]], dtype=float),
    np.array([PARTITION_OUTER[5], PARTITION_OUTER[0], PARTITION_INNER[0], PARTITION_INNER[5]], dtype=float),
]
