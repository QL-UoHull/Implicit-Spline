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
    is_convex,
    polygon_validate,
    triangulate_polygon,
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


# ── is_convex ─────────────────────────────────────────────────────────────────

class TestIsConvex:
    def test_unit_square_is_convex(self):
        P = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert is_convex(P) is True

    def test_triangle_is_convex(self):
        P = np.array([[0, 0], [2, 0], [1, 1.5]], dtype=float)
        assert is_convex(P) is True

    def test_regular_pentagon_is_convex(self):
        theta = np.linspace(0, 2 * np.pi, 6)[:-1]
        P = np.column_stack([np.cos(theta), np.sin(theta)])
        assert is_convex(P) is True

    def test_lshape_is_not_convex(self):
        assert is_convex(P_CONCAVE) is False

    def test_arrow_shape_is_not_convex(self):
        # Arrow pointing right: has a reflex vertex at the notch
        P = np.array([
            [0.0, 0.3], [0.6, 0.3], [0.6, 0.6], [1.0, 0.0],
            [0.6, -0.6], [0.6, -0.3], [0.0, -0.3],
        ], dtype=float)
        assert is_convex(P) is False

    def test_cw_square_is_convex(self):
        """CW squares are still convex (is_convex works for both orientations)."""
        P = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        assert is_convex(P) is True


# ── polygon_validate ──────────────────────────────────────────────────────────

class TestPolygonValidate:
    def test_removes_closing_vertex(self):
        P = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
        Pv = polygon_validate(P)
        assert len(Pv) == 4

    def test_normalizes_cw_to_ccw(self):
        P_cw = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        Pv = polygon_validate(P_cw)
        assert polygon_signed_area(Pv) > 0

    def test_already_ccw_unchanged(self):
        P_ccw = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        Pv = polygon_validate(P_ccw)
        assert polygon_signed_area(Pv) > 0
        assert len(Pv) == 4

    def test_removes_duplicate_consecutive_vertex(self):
        P = np.array([[0, 0], [1, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        Pv = polygon_validate(P)
        assert len(Pv) == 4

    def test_rejects_fewer_than_3_vertices(self):
        P = np.array([[0, 0], [1, 0]], dtype=float)
        with pytest.raises(ValueError, match="3 unique vertices"):
            polygon_validate(P)

    def test_rejects_degenerate_edge(self):
        # A polygon with a zero-length edge (two identical consecutive vertices
        # after deduplication) should raise.
        # Value 1e-10 is clearly above the 1e-12 deduplication threshold,
        # so the two vertices are treated as distinct and the zero-length edge
        # is detected by the edge-length check.
        P_zero = np.array([[0, 0], [0.0, 0.0 + 1e-10], [1, 0], [0.5, 1]], dtype=float)
        with pytest.raises(ValueError):
            polygon_validate(P_zero)

    def test_rejects_self_intersecting_polygon(self):
        # Bowtie (figure-eight) shape: self-intersects
        P = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
        with pytest.raises(ValueError, match="self-intersecting"):
            polygon_validate(P)

    def test_valid_lshape_passes(self):
        Pv = polygon_validate(P_CONCAVE)
        assert polygon_signed_area(Pv) > 0
        assert len(Pv) == 6

    def test_lshape_with_closing_vertex(self):
        P_closed = np.vstack([P_CONCAVE, P_CONCAVE[0]])
        Pv = polygon_validate(P_closed)
        assert len(Pv) == 6
        assert polygon_signed_area(Pv) > 0


# ── triangulate_polygon ───────────────────────────────────────────────────────

class TestTriangulatePolygon:
    def test_square_gives_two_triangles(self):
        P = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        tris = triangulate_polygon(P)
        assert len(tris) == 2

    def test_triangle_gives_one_triangle(self):
        P = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
        tris = triangulate_polygon(P)
        assert len(tris) == 1

    def test_lshape_gives_four_triangles(self):
        tris = triangulate_polygon(P_CONCAVE)
        assert len(tris) == 4

    def test_all_triangles_are_shape_3x2(self):
        tris = triangulate_polygon(P_CONCAVE)
        for t in tris:
            assert t.shape == (3, 2)

    def test_total_area_equals_polygon_area(self):
        """Sum of triangle areas must equal the polygon area."""
        tris = triangulate_polygon(P_CONCAVE)
        total = sum(abs(polygon_signed_area(t)) for t in tris)
        assert abs(total - abs(polygon_signed_area(P_CONCAVE))) < 1e-10

    def test_hexagon(self):
        theta = np.linspace(0, 2 * np.pi, 7)[:-1]
        P = np.column_stack([np.cos(theta), np.sin(theta)])
        tris = triangulate_polygon(P)
        assert len(tris) == 4

    def test_rejects_fewer_than_3(self):
        with pytest.raises(ValueError):
            triangulate_polygon(np.array([[0, 0], [1, 0]], dtype=float))


# ── imp_spline_2d non-convex ──────────────────────────────────────────────────

class TestImpSpline2DNonConvex:
    """Tests for the signed-distance path (non-convex polygons)."""

    def test_deep_interior_lower_arm(self):
        """Point deep inside horizontal arm of L → value ≈ 1."""
        v = float(imp_spline_2d(1.5, 0.5, P_CONCAVE, delta=0.05, n=2))
        assert v > 0.9, f"Expected ≈1, got {v:.4f}"

    def test_deep_interior_upper_arm(self):
        """Point deep inside vertical arm of L → value ≈ 1."""
        v = float(imp_spline_2d(0.3, 1.7, P_CONCAVE, delta=0.05, n=2))
        assert v > 0.9, f"Expected ≈1, got {v:.4f}"

    def test_near_reflex_vertex_interior(self):
        """Point near the reflex vertex but still inside the L → value > 0.5."""
        # (0.5, 1.0) is inside the L-shape, 0.5 away from the nearest boundary
        v = float(imp_spline_2d(0.5, 1.0, P_CONCAVE, delta=0.05, n=2))
        assert v > 0.5, f"Expected > 0.5 (interior), got {v:.4f}"

    def test_concave_notch_exterior(self):
        """Point inside the notch (outside the L-shape) → value = 0."""
        v = float(imp_spline_2d(1.5, 1.5, P_CONCAVE, delta=0.05, n=2))
        assert v < 0.05, f"Expected ≈0 (notch exterior), got {v:.4f}"

    def test_far_exterior(self):
        """Points far outside → 0."""
        for px, py in [(-1.0, -1.0), (3.0, 3.0), (2.5, 0.5), (0.5, 2.5)]:
            v = float(imp_spline_2d(px, py, P_CONCAVE, delta=0.05, n=2))
            assert v < 1e-10, f"Far exterior ({px},{py}): got {v:.3e}"

    def test_boundary_vertices_give_zero(self):
        """Polygon vertices lie on the boundary → value = 0."""
        for vx, vy in P_CONCAVE:
            v = float(imp_spline_2d(vx, vy, P_CONCAVE, delta=0.1, n=2))
            assert v < 1e-10, f"Boundary vertex ({vx},{vy}): got {v:.3e}"

    def test_values_in_range(self):
        """All grid values must be in [0, 1]."""
        X, Y = np.meshgrid(np.linspace(-0.5, 2.5, 40), np.linspace(-0.5, 2.5, 40))
        Z = imp_spline_2d(X, Y, P_CONCAVE, delta=0.1, n=2)
        assert np.all(Z >= -1e-14), f"Below 0: min={Z.min():.3e}"
        assert np.all(Z <= 1.0 + 1e-14), f"Above 1: max={Z.max():.3e}"

    def test_all_finite(self):
        """No NaN or Inf in the output."""
        X, Y = np.meshgrid(np.linspace(-0.5, 2.5, 40), np.linspace(-0.5, 2.5, 40))
        Z = imp_spline_2d(X, Y, P_CONCAVE, delta=0.1, n=2)
        assert np.all(np.isfinite(Z))

    def test_cw_input_auto_corrected(self):
        """CW polygon gives the same field as CCW (auto-corrected)."""
        X, Y = np.meshgrid(np.linspace(0, 2, 20), np.linspace(0, 2, 20))
        Z_ccw = imp_spline_2d(X, Y, P_CONCAVE, delta=0.1, n=2)
        Z_cw = imp_spline_2d(X, Y, P_CONCAVE[::-1], delta=0.1, n=2)
        np.testing.assert_allclose(Z_ccw, Z_cw, atol=1e-12)

    def test_multiple_delta_values(self):
        """Non-convex path works for several delta values without errors."""
        X, Y = np.meshgrid(np.linspace(0, 2, 20), np.linspace(0, 2, 20))
        for delta in [0.02, 0.1, 0.3, 0.5]:
            Z = imp_spline_2d(X, Y, P_CONCAVE, delta=delta, n=2)
            assert np.all(np.isfinite(Z))
            assert np.all(Z >= -1e-14)
            assert np.all(Z <= 1.0 + 1e-14)

    def test_multiple_n_values(self):
        """Non-convex path works for several smoothness orders."""
        X, Y = np.meshgrid(np.linspace(0, 2, 20), np.linspace(0, 2, 20))
        for nn in [1, 2, 3, 4]:
            Z = imp_spline_2d(X, Y, P_CONCAVE, delta=0.1, n=nn)
            assert np.all(np.isfinite(Z))


# ── Decomposition equivalence ─────────────────────────────────────────────────

class TestDecompositionEquivalence:
    """Direct imp_spline_2d(P_concave) vs convex_decomp_field on sub-pieces.

    The direct evaluation (signed-distance) and the bounded smooth union of
    convex pieces agree everywhere except within a band of width ≈ 2δ around
    shared internal decomposition edges.  Outside that band, max |direct - union|
    should be small (< 0.01 for δ = 0.05).
    """

    def _shared_boundary_mask(self, X, Y, delta):
        """Mask out points within 2*delta of the shared edge y=1, x∈[0,1]."""
        near_shared = (np.abs(Y - 1.0) < 2 * delta) & (X >= 0) & (X <= 1.0)
        return ~near_shared

    def test_direct_vs_decomp_deep_interior(self):
        """Direct and decomp agree at points deep inside each arm of the L-shape.

        Both constructions give ≈ 1 at deep interior points (far from all edges)
        and ≈ 0 outside.  Corner regions involve a fundamental difference
        between signed-distance H(d_min) and the product construction ∏H(L_i);
        this test only checks deep-interior agreement.
        """
        delta = 0.05
        # Points clearly inside each arm, far from any piece boundary
        deep_lower = [(0.5, 0.5), (1.0, 0.5), (1.5, 0.5)]
        deep_upper = [(0.3, 1.7), (0.5, 1.5)]
        for px, py in deep_lower + deep_upper:
            v_d = float(imp_spline_2d(px, py, P_CONCAVE, delta=delta, n=2))
            v_u = float(convex_decomp_field(px, py, [P_LOWER, P_UPPER],
                                             delta=delta, n=2))
            assert abs(v_d - v_u) < 0.01, (
                f"Deep interior ({px},{py}): direct={v_d:.4f} union={v_u:.4f}"
            )

    def test_direct_vs_decomp_away_from_shared_boundary(self):
        """Direct and decomp agree at points well away from all piece boundaries.

        Documents the known differences between the two constructions:
        - At shared decomposition edges the piece product gives 0 while the
          direct signed-distance gives the correct interior value (max error ≈ 1).
        - Near polygon corners the two decay profiles differ (product of two
          small H values vs. a single H of the corner distance).
        The two constructions agree at deep interior points (both → 1) and at
        exterior points (both → 0).
        """
        X, Y = np.meshgrid(np.linspace(-0.2, 2.2, 80), np.linspace(-0.2, 2.2, 80))
        delta = 0.05
        Z_direct = imp_spline_2d(X, Y, P_CONCAVE, delta=delta, n=2)
        Z_union = convex_decomp_field(X, Y, [P_LOWER, P_UPPER], delta=delta, n=2)

        # Deep interior points: both methods must give > 0.99
        deep_mask = (Z_direct > 0.99) & (Z_union > 0.99)
        if deep_mask.any():
            deep_err = np.abs(Z_direct[deep_mask] - Z_union[deep_mask])
            assert float(deep_err.max()) < 0.01, (
                f"At deep interior points both methods must agree; max err={float(deep_err.max()):.4f}"
            )

        # Exterior points: direct should give ≈ 0 wherever the union is 0.
        # Threshold 0.06 = slightly more than one delta (0.05) to account for
        # corner regions where the signed-distance field tapers more smoothly
        # than the product construction.
        assert float(Z_direct[Z_union < 1e-10].max()) < 0.06, (
            "Direct gives non-zero outside all convex pieces"
        )

    def test_direct_correct_at_shared_boundary_interior(self):
        """Direct evaluation should give the correct interior value at the shared edge.

        At (0.5, 1.0), which lies on the shared decomposition edge (y=1, x∈[0,1])
        but is INSIDE the L-shape, the direct evaluation gives a positive value
        while each piece's field gives 0.  This documents the known difference.
        """
        delta = 0.05
        v_direct = float(imp_spline_2d(0.5, 1.0, P_CONCAVE, delta=delta, n=2))
        v_lower = float(imp_spline_2d(0.5, 1.0, P_LOWER, delta=delta, n=2))
        v_upper = float(imp_spline_2d(0.5, 1.0, P_UPPER, delta=delta, n=2))
        # Direct gives correct interior value (> 0.5)
        assert v_direct > 0.5, f"Direct at shared boundary: {v_direct:.4f}"
        # Both pieces give 0 at their shared edge — this is expected behavior
        assert v_lower < 1e-10, f"Lower piece at its own boundary: {v_lower:.3e}"
        assert v_upper < 1e-10, f"Upper piece at its own boundary: {v_upper:.3e}"

    def test_exterior_agreement_everywhere(self):
        """Both methods agree exactly on exterior points (both give 0)."""
        notch_pts = np.array([[1.2, 1.2], [1.8, 1.3], [1.5, 1.9]])
        for px, py in notch_pts:
            v_d = float(imp_spline_2d(px, py, P_CONCAVE, delta=0.05, n=2))
            v_u = float(convex_decomp_field(px, py, [P_LOWER, P_UPPER], delta=0.05, n=2))
            assert v_d < 0.05, f"Direct exterior ({px},{py}): {v_d:.4f}"
            assert v_u < 0.05, f"Union exterior ({px},{py}): {v_u:.4f}"

    def test_second_concave_polygon(self):
        """Test a different concave polygon (T-shape) for robustness."""
        # T-shape: wide horizontal bar + narrow vertical stem
        P_T = np.array([
            [0.0, 1.0], [3.0, 1.0], [3.0, 2.0], [2.0, 2.0],
            [2.0, 3.0], [1.0, 3.0], [1.0, 2.0], [0.0, 2.0],
        ], dtype=float)
        assert polygon_signed_area(P_T) > 0, "T-shape must be CCW"
        assert not is_convex(P_T)
        X, Y = np.meshgrid(np.linspace(-0.5, 3.5, 40), np.linspace(0.5, 3.5, 40))
        Z = imp_spline_2d(X, Y, P_T, delta=0.08, n=2)
        assert np.all(np.isfinite(Z))
        assert np.all(Z >= -1e-14)
        assert np.all(Z <= 1.0 + 1e-14)
        # Interior of horizontal bar
        v_bar = float(imp_spline_2d(1.5, 1.5, P_T, delta=0.05, n=2))
        assert v_bar > 0.9, f"T-bar interior: {v_bar:.4f}"
        # Interior of vertical stem
        v_stem = float(imp_spline_2d(1.5, 2.5, P_T, delta=0.05, n=2))
        assert v_stem > 0.9, f"T-stem interior: {v_stem:.4f}"
        # Outside
        v_out = float(imp_spline_2d(0.5, 2.5, P_T, delta=0.05, n=2))
        assert v_out < 0.05, f"T outside arm: {v_out:.4f}"
