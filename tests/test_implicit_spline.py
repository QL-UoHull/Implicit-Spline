"""
tests/test_implicit_spline.py
-----------------------------
Tests for the corrected concave-polygon and partition-basis functionality.

Covers:
- CCW orientation of the canonical concave (L-shaped) polygon fixture.
- Bounded smooth union: range in [0, 1] and identity properties.
- convex_decomp_field: field bounds, interior/exterior values,
  and correct representation of the L-shaped concave polygon.
- partition_basis_normalized: finite values, range, partition-of-unity,
  and normalization at cell boundaries.
- Safe contour-level handling in visualization helpers.
"""

import sys
import os

# Ensure the package is importable from both the repo root and tests/
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.join(_tests_dir, '..', 'python')
if os.path.isdir(_python_dir):
    sys.path.insert(0, os.path.abspath(_python_dir))

import numpy as np
import pytest

from implicit_spline.core import (
    polygon_signed_area,
    smooth_union,
    convex_decomp_field,
    partition_basis_normalized,
    imp_spline_2d,
)

# ── Thresholds ────────────────────────────────────────────────────────────────
#: Minimum expected field value well inside the shape (for delta=0.05).
INTERIOR_THRESHOLD = 0.5
#: Maximum expected field value well outside the shape / in the concave notch.
EXTERIOR_THRESHOLD = 0.05

# ── Canonical fixtures ────────────────────────────────────────────────────────

# L-shaped concave polygon (vertices in CCW boundary order).
#
#   (0,2)──(1,2)
#   |       |
#   |  upper|
#   |       |
#   (0,1)──(1,1)──(2,1)
#   |               |
#   |     lower     |
#   |               |
#   (0,0)──────────(2,0)
#
P_CONCAVE = np.array(
    [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]],
    dtype=float,
)

# Convex decomposition of P_CONCAVE into two non-overlapping rectangles.
P_LOWER = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
P_UPPER = np.array([[0, 1], [1, 1], [1, 2], [0, 2]], dtype=float)

# Simple 4-cell rectangular partition of [0, 2] × [0, 1].
CELLS_2X2 = [
    np.array([[0, 0], [1, 0], [1, 0.5], [0, 0.5]], dtype=float),
    np.array([[1, 0], [2, 0], [2, 0.5], [1, 0.5]], dtype=float),
    np.array([[0, 0.5], [1, 0.5], [1, 1], [0, 1]], dtype=float),
    np.array([[1, 0.5], [2, 0.5], [2, 1], [1, 1]], dtype=float),
]


# ── polygon_signed_area / CCW orientation ─────────────────────────────────────

class TestPolygonSignedArea:
    def test_concave_polygon_is_ccw(self):
        """P_CONCAVE must be in CCW order (positive signed area)."""
        assert polygon_signed_area(P_CONCAVE) > 0

    def test_l_shape_area_exact(self):
        """L-shape is a 2×2 square minus a 1×1 notch → area = 3."""
        assert abs(polygon_signed_area(P_CONCAVE) - 3.0) < 1e-12

    def test_cw_polygon_negative_area(self):
        """CW-ordered polygon has negative signed area."""
        assert polygon_signed_area(P_CONCAVE[::-1]) < 0

    def test_unit_square_ccw(self):
        """CCW unit square has area 1."""
        P = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert abs(polygon_signed_area(P) - 1.0) < 1e-12


# ── smooth_union ──────────────────────────────────────────────────────────────

class TestSmoothUnion:
    def test_range_random(self):
        """smooth_union of random [0,1] inputs must stay in [0,1]."""
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 1, (50, 50))
        b = rng.uniform(0, 1, (50, 50))
        result = smooth_union(a, b)
        assert np.all(result >= 0.0), "smooth_union returned negative values"
        assert np.all(result <= 1.0), "smooth_union returned values > 1"

    def test_identity_zero_right(self):
        """smooth_union(0, b) == b for any b."""
        b = np.array([0.0, 0.3, 0.7, 1.0])
        np.testing.assert_allclose(smooth_union(np.zeros_like(b), b), b)

    def test_identity_zero_left(self):
        """smooth_union(a, 0) == a for any a."""
        a = np.array([0.0, 0.3, 0.7, 1.0])
        np.testing.assert_allclose(smooth_union(a, np.zeros_like(a)), a)

    def test_one_absorbs(self):
        """smooth_union(1, b) == 1 for any b in [0,1]."""
        b = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(smooth_union(np.ones_like(b), b), np.ones_like(b))

    def test_symmetry(self):
        """smooth_union(a, b) == smooth_union(b, a)."""
        rng = np.random.default_rng(1)
        a = rng.uniform(0, 1, (20,))
        b = rng.uniform(0, 1, (20,))
        np.testing.assert_allclose(smooth_union(a, b), smooth_union(b, a))


# ── convex_decomp_field ───────────────────────────────────────────────────────

class TestConvexDecompField:
    def test_bounds_over_grid(self):
        """convex_decomp_field must return values in [0, 1] over a full grid."""
        X, Y = np.meshgrid(np.linspace(-0.5, 2.5, 50), np.linspace(-0.5, 2.5, 50))
        Z = convex_decomp_field(X, Y, [P_LOWER, P_UPPER], delta=0.1, n=2)
        assert np.all(Z >= -1e-14), f"Field below 0: min={Z.min():.3e}"
        assert np.all(Z <= 1.0 + 1e-14), f"Field above 1: max={Z.max():.3e}"

    def test_interior_lower_piece(self):
        """Deep interior of lower piece should have high field value."""
        Z = float(convex_decomp_field(0.5, 0.5, [P_LOWER, P_UPPER], delta=0.05, n=2))
        assert Z > INTERIOR_THRESHOLD, (
            f"Interior of lower piece: Z={Z:.4f}, expected > {INTERIOR_THRESHOLD}"
        )

    def test_interior_upper_piece(self):
        """Deep interior of upper piece should have high field value."""
        Z = float(convex_decomp_field(0.3, 1.6, [P_LOWER, P_UPPER], delta=0.05, n=2))
        assert Z > INTERIOR_THRESHOLD, (
            f"Interior of upper piece: Z={Z:.4f}, expected > {INTERIOR_THRESHOLD}"
        )

    def test_concave_notch_exterior(self):
        """Point in the concave notch (1.5, 1.5) is outside the L-shape: Z ≈ 0."""
        Z = float(convex_decomp_field(1.5, 1.5, [P_LOWER, P_UPPER], delta=0.05, n=2))
        assert Z < EXTERIOR_THRESHOLD, (
            f"Concave notch point: Z={Z:.4f}, expected < {EXTERIOR_THRESHOLD}"
        )

    def test_far_exterior(self):
        """Points far outside the shape should have field ≈ 0."""
        Z = float(convex_decomp_field(-2.0, -2.0, [P_LOWER, P_UPPER], delta=0.1, n=2))
        assert Z < 1e-12, f"Far exterior: Z={Z:.3e}"

    def test_single_piece_matches_imp_spline(self):
        """convex_decomp_field with one piece equals imp_spline_2d."""
        X, Y = np.meshgrid(np.linspace(0, 2, 30), np.linspace(0, 1, 30))
        Z_decomp = convex_decomp_field(X, Y, [P_LOWER], delta=0.1, n=2)
        Z_direct = imp_spline_2d(X, Y, P_LOWER, delta=0.1, n=2)
        np.testing.assert_allclose(Z_decomp, Z_direct, atol=1e-12)

    def test_l_shape_interior_points(self):
        """Test that representative interior points of the L-shaped P_CONCAVE have
        high field values when evaluated via its convex decomposition."""
        decomp = [P_LOWER, P_UPPER]
        delta = 0.05
        # Deep inside the horizontal arm of the L
        Z_horiz = float(convex_decomp_field(1.5, 0.5, decomp, delta=delta, n=2))
        assert Z_horiz > INTERIOR_THRESHOLD, (
            f"L-shape horizontal arm: Z={Z_horiz:.4f}, expected > {INTERIOR_THRESHOLD}"
        )
        # Deep inside the vertical arm of the L
        Z_vert = float(convex_decomp_field(0.3, 1.7, decomp, delta=delta, n=2))
        assert Z_vert > INTERIOR_THRESHOLD, (
            f"L-shape vertical arm: Z={Z_vert:.4f}, expected > {INTERIOR_THRESHOLD}"
        )
        # In the concave notch (outside the L-shape)
        Z_notch = float(convex_decomp_field(1.5, 1.5, decomp, delta=delta, n=2))
        assert Z_notch < EXTERIOR_THRESHOLD, (
            f"L-shape notch (exterior): Z={Z_notch:.4f}, expected < {EXTERIOR_THRESHOLD}"
        )


# ── partition_basis_normalized ────────────────────────────────────────────────

class TestPartitionBasisNormalized:
    def test_returns_correct_count(self):
        """Returns one basis array per polygon."""
        X, Y = np.meshgrid(np.linspace(0, 2, 10), np.linspace(0, 1, 10))
        basis, raw_sum = partition_basis_normalized(CELLS_2X2, X, Y, delta=0.1, n=2)
        assert len(basis) == len(CELLS_2X2)

    def test_all_finite(self):
        """All basis values must be finite (no NaN or Inf)."""
        X, Y = np.meshgrid(np.linspace(-0.5, 2.5, 40), np.linspace(-0.5, 1.5, 40))
        basis, _ = partition_basis_normalized(CELLS_2X2, X, Y, delta=0.1, n=2)
        for k, b in enumerate(basis):
            assert np.all(np.isfinite(b)), f"Cell {k} has non-finite values"

    def test_range(self):
        """Normalized basis values must lie in [0, 1]."""
        X, Y = np.meshgrid(np.linspace(-0.5, 2.5, 40), np.linspace(-0.5, 1.5, 40))
        basis, _ = partition_basis_normalized(CELLS_2X2, X, Y, delta=0.1, n=2)
        for k, b in enumerate(basis):
            assert np.all(b >= -1e-14), f"Cell {k} has values below 0"
            assert np.all(b <= 1.0 + 1e-14), f"Cell {k} has values above 1"

    def test_partition_of_unity_exact(self):
        """Normalized basis sums to exactly 1 at any point with raw_sum > eps."""
        X, Y = np.meshgrid(np.linspace(0.2, 1.8, 40), np.linspace(0.1, 0.9, 40))
        basis, raw_sum = partition_basis_normalized(CELLS_2X2, X, Y, delta=0.15, n=2)
        total = sum(basis)
        active = raw_sum > 1e-9
        if active.any():
            max_err = float(np.max(np.abs(total[active] - 1.0)))
            assert max_err < 1e-10, (
                f"Partition of unity violated: max|sum-1|={max_err:.3e}"
            )

    def test_partition_of_unity_at_shared_boundary(self):
        """Normalization must give sum == 1 at points on shared cell edges."""
        # The shared boundary between top and bottom cells is at y=0.5.
        # Pick x=0.5, y=0.5 which lies on the shared edge between cells 0 and 2.
        x_pt = np.array([0.5])
        y_pt = np.array([0.5])
        basis, raw_sum = partition_basis_normalized(CELLS_2X2, x_pt, y_pt,
                                                    delta=0.15, n=2)
        total = float(sum(b[0] for b in basis))
        rs = float(raw_sum[0])
        if rs > 1e-9:
            assert abs(total - 1.0) < 1e-10, (
                f"Partition sum at shared boundary: {total:.6f}, expected 1.0"
            )

    def test_raw_sum_non_negative(self):
        """Raw sum of cell fields must be non-negative everywhere."""
        X, Y = np.meshgrid(np.linspace(0, 2, 30), np.linspace(0, 1, 30))
        _, raw_sum = partition_basis_normalized(CELLS_2X2, X, Y, delta=0.1, n=2)
        assert np.all(raw_sum >= 0), "Raw sum has negative values"

    def test_exterior_points_zero(self):
        """Points far outside all cells should have near-zero basis values."""
        basis, raw_sum = partition_basis_normalized(
            CELLS_2X2,
            np.array([-5.0]),
            np.array([-5.0]),
            delta=0.1,
            n=2,
        )
        for k, b in enumerate(basis):
            assert float(b[0]) < 1e-12, f"Cell {k} far exterior: {float(b[0]):.3e}"


# ── safe contour level handling ───────────────────────────────────────────────

class TestSafeContour:
    """Verify that _safe_contour does not raise when levels are outside range."""

    def setup_method(self):
        import matplotlib
        matplotlib.use('Agg')

    def test_does_not_raise_when_in_range(self):
        import matplotlib.pyplot as plt
        from implicit_spline.visualization import _safe_contour
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        Z = X * Y  # range [0, 1]
        _safe_contour(ax, X, Y, Z, levels=[0.5], colors='k')
        plt.close('all')

    def test_does_not_raise_when_outside_range(self):
        """No error when iso_level exceeds Z.max()."""
        import matplotlib.pyplot as plt
        from implicit_spline.visualization import _safe_contour
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        Z = np.zeros_like(X)  # all zeros; 0.5 is outside (0, 0)
        _safe_contour(ax, X, Y, Z, levels=[0.5], colors='k')
        plt.close('all')

    def test_draw_imp_spline_no_raise(self):
        """draw_imp_spline must not raise regardless of iso_level choice."""
        import matplotlib.pyplot as plt
        from implicit_spline.visualization import draw_imp_spline
        P = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        # Large delta: Z may never reach 0.99
        draw_imp_spline(P, delta=0.8, n=2, N=30, iso_level=0.99)
        plt.close('all')

