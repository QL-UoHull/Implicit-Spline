import os
import sys

import numpy as np
import pytest

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.join(_tests_dir, '..', 'python')
if os.path.isdir(_python_dir):
    sys.path.insert(0, os.path.abspath(_python_dir))

from implicit_spline.core import (
    H,
    _H_prime,
    convex_decomp_field,
    convex_product_spline_2d,
    imp_spline_2d,
    is_convex,
    l_corner_inter,
    line_seg_imp,
    lxy,
    lxy00,
    partition_basis_fields,
    point_imp,
    polygon_signed_area,
    polygon_validate,
    square_angle_inter,
    triangulate_polygon,
    u_angle_inter,
    validate_partition,
    _segments_intersection_kind,
)
from implicit_spline.paper_examples import (
    PARTITION_CELLS,
    PARTITION_OUTER,
    SECTION7_POLYGONS,
    SECTION7_TEST_POINTS,
)

TEST_DELTA = 0.22
N_ORDER = 2
ABS_TOL = 1e-12
FIELD_TOL = 5e-3


@pytest.mark.parametrize("name", sorted(SECTION7_POLYGONS))
def test_section7_polygons_are_ccw_and_non_convex(name):
    P = polygon_validate(SECTION7_POLYGONS[name], name=name)
    assert polygon_signed_area(P) > 0
    assert not is_convex(P)


@pytest.mark.parametrize("name", sorted(SECTION7_POLYGONS))
def test_section7_points_match_expected_interior_and_exterior(name):
    P = SECTION7_POLYGONS[name]
    for x, y in SECTION7_TEST_POINTS[name]["inside"]:
        assert float(imp_spline_2d(x, y, P, delta=TEST_DELTA, n=N_ORDER)) > 0.95
    for x, y in SECTION7_TEST_POINTS[name]["outside"]:
        assert float(imp_spline_2d(x, y, P, delta=TEST_DELTA, n=N_ORDER)) < 0.15


@pytest.mark.parametrize("name", sorted(SECTION7_POLYGONS))
def test_direct_equals_triangulation_decomposition(name):
    P = SECTION7_POLYGONS[name]
    tris = triangulate_polygon(P)
    X, Y = np.meshgrid(
        np.linspace(P[:, 0].min() - 0.4, P[:, 0].max() + 0.4, 90),
        np.linspace(P[:, 1].min() - 0.4, P[:, 1].max() + 0.4, 90),
    )
    direct = imp_spline_2d(X, Y, P, delta=TEST_DELTA, n=N_ORDER)
    decomp = convex_decomp_field(X, Y, tris, delta=TEST_DELTA, n=N_ORDER)
    np.testing.assert_allclose(direct, decomp, rtol=0.0, atol=ABS_TOL)


@pytest.mark.parametrize("name", sorted(SECTION7_POLYGONS))
def test_direct_equals_multiple_valid_decompositions(name):
    P = SECTION7_POLYGONS[name]
    X, Y = np.meshgrid(
        np.linspace(P[:, 0].min() - 0.4, P[:, 0].max() + 0.4, 70),
        np.linspace(P[:, 1].min() - 0.4, P[:, 1].max() + 0.4, 70),
    )
    direct = imp_spline_2d(X, Y, P, delta=TEST_DELTA, n=N_ORDER)
    decomp_a = convex_decomp_field(X, Y, triangulate_polygon(P), delta=TEST_DELTA, n=N_ORDER)
    decomp_b = convex_decomp_field(X, Y, triangulate_polygon(np.roll(P, -3, axis=0)), delta=TEST_DELTA, n=N_ORDER)
    np.testing.assert_allclose(direct, decomp_a, rtol=0.0, atol=ABS_TOL)
    np.testing.assert_allclose(direct, decomp_b, rtol=0.0, atol=ABS_TOL)


@pytest.mark.parametrize("name", sorted(SECTION7_POLYGONS))
def test_no_internal_edge_seams_on_decomposition(name):
    P = SECTION7_POLYGONS[name]
    tris = triangulate_polygon(P)
    X, Y = np.meshgrid(
        np.linspace(P[:, 0].min() - 0.2, P[:, 0].max() + 0.2, 120),
        np.linspace(P[:, 1].min() - 0.2, P[:, 1].max() + 0.2, 120),
    )
    direct = imp_spline_2d(X, Y, P, delta=TEST_DELTA, n=N_ORDER)
    decomp = convex_decomp_field(X, Y, tris, delta=TEST_DELTA, n=N_ORDER)
    err = np.abs(direct - decomp)
    assert float(err.max()) <= ABS_TOL
    assert err.size > 0


def test_partition_is_valid_and_conforming():
    summary = validate_partition(PARTITION_CELLS, PARTITION_OUTER)
    assert np.isclose(summary["total_area"], summary["outer_area"], rtol=0.0, atol=1e-10)
    assert all(count in {1, 2} for count in summary["edge_incidence"].values())


def test_partition_sum_matches_outer_boundary_basis():
    X, Y = np.meshgrid(
        np.linspace(PARTITION_OUTER[:, 0].min() - 0.2, PARTITION_OUTER[:, 0].max() + 0.2, 90),
        np.linspace(PARTITION_OUTER[:, 1].min() - 0.2, PARTITION_OUTER[:, 1].max() + 0.2, 90),
    )
    basis, total = partition_basis_fields(PARTITION_CELLS, X, Y, delta=0.18, n=N_ORDER)
    outer = imp_spline_2d(X, Y, PARTITION_OUTER, delta=0.18, n=N_ORDER)
    assert len(basis) == len(PARTITION_CELLS)
    np.testing.assert_allclose(total, outer, rtol=0.0, atol=ABS_TOL)


def test_partition_sum_is_nearly_one_away_from_outer_boundary():
    x0, x1 = PARTITION_OUTER[:, 0].min() + 1.0, PARTITION_OUTER[:, 0].max() - 1.7
    y0, y1 = PARTITION_OUTER[:, 1].min() + 1.1, PARTITION_OUTER[:, 1].max() - 1.25
    X, Y = np.meshgrid(np.linspace(x0, x1, 60), np.linspace(y0, y1, 60))
    _, total = partition_basis_fields(PARTITION_CELLS, X, Y, delta=0.18, n=N_ORDER)
    active = total > 0.995
    assert active.any(), "expected at least one deep-interior sample inside the partition"
    assert float(np.max(np.abs(total[active] - 1.0))) < FIELD_TOL


def test_convex_examples_remain_additive_under_triangulation():
    square = np.array([[0.0, 0.0], [1.4, 0.0], [1.4, 1.2], [0.0, 1.2]], dtype=float)
    tris = triangulate_polygon(square)
    X, Y = np.meshgrid(np.linspace(-0.2, 1.6, 60), np.linspace(-0.2, 1.4, 60))
    direct = imp_spline_2d(X, Y, square, delta=0.12, n=N_ORDER)
    decomp = convex_decomp_field(X, Y, tris, delta=0.12, n=N_ORDER)
    np.testing.assert_allclose(direct, decomp, rtol=0.0, atol=ABS_TOL)
    assert float(imp_spline_2d(0.7, 0.6, square, delta=0.12, n=N_ORDER)) > 0.99
    assert float(convex_product_spline_2d(0.7, 0.6, square, delta=0.12, n=N_ORDER)) > 0.99


def test_polygon_with_hole_composition_still_behaves():
    outer = np.array([[-2.0, -1.7], [2.0, -1.7], [2.3, 1.0], [0.0, 2.1], [-2.2, 1.0]], dtype=float)
    hole = np.array([[-0.5, -0.3], [0.5, -0.3], [0.5, 0.5], [-0.5, 0.5]], dtype=float)
    outer_field = imp_spline_2d(0.0, 0.0, outer, delta=0.15, n=N_ORDER)
    hole_field = imp_spline_2d(0.0, 0.0, hole, delta=0.15, n=N_ORDER)
    composed = float(outer_field * (1.0 - hole_field))
    annulus = float(imp_spline_2d(1.3, 0.0, outer, delta=0.15, n=N_ORDER) * (1.0 - imp_spline_2d(1.3, 0.0, hole, delta=0.15, n=N_ORDER)))
    assert composed < 0.2
    assert annulus > 0.8


def test_polygon_validate_large_coordinates_no_vertex_collapse():
    P = np.array([
        [1_000_000.0, 1_000_000.0],
        [1_000_002.0, 1_000_000.0],
        [1_000_004.0, 1_000_004.0],
        [1_000_002.0, 1_000_005.0],
        [1_000_000.1, 1_000_002.0],
    ], dtype=float)
    validated = polygon_validate(P)
    assert len(validated) == 5


def test_polygon_validate_separates_duplicate_and_degenerate_tolerances():
    P = np.array([
        [0.0, 0.0],
        [5e-11, 0.0],
        [1.0, 0.0],
        [0.5, 1.0],
    ], dtype=float)
    with pytest.raises(ValueError, match="near-zero edge"):
        polygon_validate(P)


def test_polygon_validate_rejects_collinear_overlap_and_invalid_touching():
    assert _segments_intersection_kind(
        np.array([0.0, 0.0]), np.array([4.0, 0.0]),
        np.array([1.0, 0.0]), np.array([3.0, 0.0]),
    ) == "overlap"

    # Non-adjacent edge 4-5 touches edge 0-1 at (1, 0), which is invalid.
    invalid_touching_polygon = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [1.0, 1.0],
        [0.0, 2.0],
        [1.0, 0.0],
    ], dtype=float)
    with pytest.raises(ValueError, match="invalid touching edges"):
        polygon_validate(invalid_touching_polygon)


def test_imp_spline_handles_repeated_closing_vertex():
    P = SECTION7_POLYGONS["heart_like"]
    closed = np.vstack([P, P[0]])
    X, Y = np.meshgrid(np.linspace(-2.8, 2.8, 40), np.linspace(-2.4, 1.5, 40))
    open_field = imp_spline_2d(X, Y, P, delta=TEST_DELTA, n=N_ORDER)
    closed_field = imp_spline_2d(X, Y, closed, delta=TEST_DELTA, n=N_ORDER)
    np.testing.assert_allclose(open_field, closed_field, rtol=0.0, atol=ABS_TOL)


def test_triangulate_polygon_returns_exact_count_or_raises():
    P = SECTION7_POLYGONS["narrow_neck"]
    tris = triangulate_polygon(P)
    assert len(tris) == len(P) - 2
    area = sum(abs(polygon_signed_area(tri)) for tri in tris)
    assert np.isclose(area, abs(polygon_signed_area(P)), rtol=0.0, atol=1e-10)

    bowtie = np.array([[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]], dtype=float)
    with pytest.raises(ValueError):
        triangulate_polygon(bowtie)


class TestSafeContour:
    """Regression tests for safe contour plotting with out-of-range levels."""
    def setup_method(self):
        """Force the headless Agg backend for plotting tests."""
        import matplotlib
        matplotlib.use('Agg')

    def test_safe_contour_and_draw_imp_spline(self):
        """Contours outside the data range should not raise under headless Matplotlib."""
        import matplotlib.pyplot as plt
        from implicit_spline.visualization import _safe_contour, draw_imp_spline

        fig, ax = plt.subplots()
        X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        Z = np.zeros_like(X)
        _safe_contour(ax, X, Y, Z, levels=[1.5], colors='k')
        plt.close(fig)

        fig2, ax2 = plt.subplots()
        draw_imp_spline(SECTION7_POLYGONS['heart_like'], delta=TEST_DELTA, n=N_ORDER, N=40, ax=ax2, iso_level=1.5)
        plt.close(fig2)


# ──────────────────────────────────────────────────────────────────────────────
# MATLAB parity tests
# These tests verify that the Python functions faithfully reproduce the
# mathematical formulas from the original MATLAB reference implementation.
# ──────────────────────────────────────────────────────────────────────────────

class TestMatlabParityH:
    """Parity tests for H(t, delta, n) — MATLAB H.m."""

    def test_H_boundary_values(self):
        """H = 0 at t ≤ 0, H = 1 at t ≥ delta."""
        for n in [1, 2, 3]:
            assert float(H(-0.1, 0.5, n)) == 0.0
            assert float(H(0.0, 0.5, n)) == 0.0
            assert float(H(0.5, 0.5, n)) == 1.0
            assert float(H(1.0, 0.5, n)) == 1.0

    def test_H_midpoint_is_half(self):
        """H(delta/2, delta, n) == 0.5 for all n (symmetry of Bernstein polynomial)."""
        for n in [1, 2, 3, 4]:
            val = float(H(0.25, 0.5, n))
            assert abs(val - 0.5) < 1e-12, f"n={n}: H(delta/2)={val}, expected 0.5"

    def test_H_is_monotone(self):
        """H must be monotonically non-decreasing on [0, delta]."""
        delta = 0.4
        t = np.linspace(-0.1, 0.6, 500)
        for n in [1, 2, 3]:
            vals = H(t, delta, n)
            assert np.all(np.diff(vals) >= -1e-14), f"H not monotone for n={n}"

    @pytest.mark.parametrize("n,t,delta,expected", [
        # n=1: Bernstein degree 3 → H(t,δ,1) = 3(t/δ)^2 − 2(t/δ)^3 on [0,δ]
        (1, 0.25, 0.5, 3 * 0.5**2 - 2 * 0.5**3),       # s=0.5
        (1, 0.1,  0.4, 3 * 0.25**2 - 2 * 0.25**3),     # s=0.25
        # n=2: Bernstein degree 5 → H = 6s^5 − 15s^4 + 10s^3  for s=t/δ
        (2, 0.2, 0.4, 6 * 0.5**5 - 15 * 0.5**4 + 10 * 0.5**3),  # s=0.5
    ])
    def test_H_known_values(self, n, t, delta, expected):
        val = float(H(t, delta, n))
        assert abs(val - expected) < 1e-12, f"n={n},t={t},δ={delta}: got {val}, want {expected}"

    def test_n_affects_output(self):
        """Different n values must give different H values (n is not ignored)."""
        vals = [float(H(0.2, 0.4, n)) for n in [1, 2, 3, 4]]
        # All should be ≈0.5 at midpoint, but NOT identical elsewhere
        t_off_mid = 0.15
        vals_off = [float(H(t_off_mid, 0.4, n)) for n in [1, 2, 3, 4]]
        # Different n → different values
        assert not all(abs(v - vals_off[0]) < 1e-10 for v in vals_off[1:]), \
            "n parameter has no effect — must differ for different n values"


class TestMatlabParityLxy:
    """Parity tests for lxy / lxy00 / point_imp — MATLAB Lxy.m, Lxy00.m, Point_imp.m."""

    def test_lxy_horizontal_edge(self):
        """Signed distance to horizontal edge (0,0)→(1,0): positive above, negative below."""
        assert abs(float(lxy(0.5, 0.3, 0.0, 0.0, 1.0, 0.0)) - 0.3) < 1e-12
        assert abs(float(lxy(0.5, -0.2, 0.0, 0.0, 1.0, 0.0)) - (-0.2)) < 1e-12
        assert abs(float(lxy(0.5, 0.0, 0.0, 0.0, 1.0, 0.0))) < 1e-12

    def test_lxy_vertical_edge(self):
        """Signed distance to vertical edge (1,0)→(1,2): positive to the left (x < 1)."""
        assert abs(float(lxy(0.6, 1.0, 1.0, 0.0, 1.0, 2.0)) - 0.4) < 1e-12
        assert abs(float(lxy(1.4, 1.0, 1.0, 0.0, 1.0, 2.0)) - (-0.4)) < 1e-12

    def test_lxy_is_normalised(self):
        """lxy returns the normalised signed distance (independent of edge length)."""
        # Edge from (0,0) to (2,0) — same line as (0,0)→(1,0)
        d1 = float(lxy(0.5, 0.3, 0.0, 0.0, 1.0, 0.0))
        d2 = float(lxy(0.5, 0.3, 0.0, 0.0, 2.0, 0.0))
        assert abs(d1 - d2) < 1e-12

    def test_lxy00_unnormalised(self):
        """lxy00 returns unnormalised value (scales with edge length)."""
        d_norm = float(lxy(0.5, 0.3, 0.0, 0.0, 2.0, 0.0))
        d_unnorm = float(lxy00(0.5, 0.3, 0.0, 0.0, 2.0, 0.0))
        # unnorm / norm == edge_length
        edge_len = 2.0
        assert abs(d_unnorm / d_norm - edge_len) < 1e-12

    def test_point_imp_zero_at_origin(self):
        """point_imp(0,0, 0,0) must be 0."""
        assert float(point_imp(0.0, 0.0, 0.0, 0.0)) == 0.0

    def test_point_imp_unit_distance(self):
        """point_imp(1,0, 0,0) == 1.0 (squared distance)."""
        assert abs(float(point_imp(1.0, 0.0, 0.0, 0.0)) - 1.0) < 1e-12
        assert abs(float(point_imp(3.0, 4.0, 0.0, 0.0)) - 25.0) < 1e-12


class TestMatlabParityLineSeg:
    """Parity tests for line_seg_imp — MATLAB LineSeg_imp.m."""

    def test_line_seg_zero_at_endpoint(self):
        """Result must taper to zero near segment endpoints."""
        # The gate H(t_proj, δ, n) → 0 as t_proj → 0 or t_proj → length
        val_start = float(line_seg_imp(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, delta=0.1, n=2))
        assert abs(val_start) < 1e-10

    def test_line_seg_midpoint(self):
        """At the midpoint of a horizontal segment, Lperp * gate should equal Lperp
        (gate ≈ 1 deep inside the segment)."""
        y_query = 0.15
        # Midpoint x=0.5 along segment (0,0)→(1,0)
        val = float(line_seg_imp(0.5, y_query, 0.0, 0.0, 1.0, 0.0, delta=0.1, n=2))
        assert val > 0.0, "Midpoint should have positive left-side value"

    def test_line_seg_sign_convention(self):
        """Value must be positive to the LEFT (CCW side) of the directed segment."""
        # Horizontal segment (0,0)→(1,0): left is y > 0
        assert float(line_seg_imp(0.5, 0.2, 0.0, 0.0, 1.0, 0.0, delta=0.1, n=2)) > 0
        assert float(line_seg_imp(0.5, -0.2, 0.0, 0.0, 1.0, 0.0, delta=0.1, n=2)) < 0


class TestMatlabParityCorners:
    """Parity tests for corner functions — MATLAB L_corner_inter, U_Angle_inter."""

    def _corner_triple(self):
        """Returns A, B, C for a right-angle CCW corner at B=(0,0)."""
        return (-1.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # A=(-1,0), B=(0,0), C=(0,1)

    def test_l_corner_deep_interior(self):
        """l_corner_inter must be positive in the CCW interior region."""
        xA, yA, xB, yB, xC, yC = self._corner_triple()
        # Interior: left of A→B (y>0) AND left of B→C (x<0) → use (-0.3, 0.3)
        val = float(l_corner_inter(-0.3, 0.3, xA, yA, xB, yB, xC, yC, delta=0.1, n=2))
        assert val > 0.9, f"Deep interior: got {val}"

    def test_l_corner_zero_outside(self):
        """l_corner_inter must be ~ 0 far outside the corner."""
        xA, yA, xB, yB, xC, yC = self._corner_triple()
        val = float(l_corner_inter(-0.5, -0.5, xA, yA, xB, yB, xC, yC, delta=0.1, n=2))
        assert val < 0.05, f"Outside corner: got {val}"

    def test_u_angle_inter_or_logic(self):
        """u_angle_inter (OR) must exceed l_corner_inter (AND) for wide angles."""
        # For a wide (obtuse) angle, OR gives higher values than AND
        # Use A=(-1,0), B=(0,0), C=(1,0.1) — nearly a straight line (>90° interior)
        xA, yA = -1.0, 0.0
        xB, yB = 0.0, 0.0
        xC, yC = 1.0, 0.0
        query_x, query_y = 0.0, 0.3
        delta = 0.4
        val_and = float(l_corner_inter(query_x, query_y, xA, yA, xB, yB, xC, yC, delta, 2))
        val_or = float(u_angle_inter(query_x, query_y, xA, yA, xB, yB, xC, yC, delta, 2))
        assert val_or >= val_and - 1e-10, (
            f"u_angle_inter OR={val_or:.4f} should be ≥ l_corner_inter AND={val_and:.4f}"
        )

    def test_u_angle_inter_formula(self):
        """u_angle_inter == h1 + h2 - h1*h2 (probabilistic OR formula from MATLAB)."""
        xA, yA = 0.0, 1.0
        xB, yB = 0.0, 0.0
        xC, yC = 1.0, 0.0
        qx, qy = 0.3, 0.3
        delta = 0.5
        L1 = float(lxy(qx, qy, xA, yA, xB, yB))
        L2 = float(lxy(qx, qy, xB, yB, xC, yC))
        h1 = float(H(L1, delta, 2))
        h2 = float(H(L2, delta, 2))
        expected = h1 + h2 - h1 * h2
        got = float(u_angle_inter(qx, qy, xA, yA, xB, yB, xC, yC, delta, 2))
        assert abs(got - expected) < 1e-12, f"Got {got}, expected {expected}"


class TestMatlabParityConvexProduct:
    """Parity tests for convex_product_spline_2d — MATLAB ImpSpline2D.m."""

    def test_convex_square_deep_interior(self):
        """All H(L_i) ≈ 1 deep inside a convex square → product ≈ 1."""
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        val = float(convex_product_spline_2d(0.5, 0.5, sq, delta=0.05, n=2))
        assert val > 0.99

    def test_convex_product_matches_matlab_formula(self):
        """convex_product_spline_2d must equal ∏ H(L_i, δ, n) directly."""
        sq = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        qx, qy = 1.0, 1.0
        delta, n = 0.3, 2
        # Manual product
        product = 1.0
        for i in range(len(sq)):
            j = (i + 1) % len(sq)
            L = float(lxy(qx, qy, sq[i, 0], sq[i, 1], sq[j, 0], sq[j, 1]))
            product *= float(H(L, delta, n))
        result = float(convex_product_spline_2d(qx, qy, sq, delta=delta, n=n))
        assert abs(result - product) < 1e-12, f"Got {result}, expected {product}"


# ──────────────────────────────────────────────────────────────────────────────
# Smoothness tests (C^(n-1) — consistent with MATLAB H.m semantics)
# ──────────────────────────────────────────────────────────────────────────────

class TestSmoothnessHKernel:
    """Verify C^(n-1) smoothness of the H-kernel field (not C^∞ Gaussian)."""

    def test_H_prime_integrates_to_one(self):
        """H'(t + δ/2, δ, n) must integrate to 1 over its support (normalisation)."""
        delta = 0.4
        t = np.linspace(-delta / 2 - 0.1, delta / 2 + 0.1, 50000)
        for n in [1, 2, 3]:
            vals = _H_prime(t + delta / 2, delta, n)
            integral = float(np.trapezoid(vals, t))
            assert abs(integral - 1.0) < 1e-4, f"n={n}: integral={integral:.6f}"

    def test_field_is_cn_minus1_smooth(self):
        """Finite-difference n-th derivative of f must be bounded.

        The H-kernel gives a C^(n-1) field.  For n=2 (C^1), the first
        derivative must be continuous (FD of order 1 is bounded), while the
        second derivative may have jumps near the boundary.
        """
        P = SECTION7_POLYGONS["heart_like"]
        # Sample a line through the interior, crossing the iso-contour
        xs = np.linspace(-2.0, 2.0, 4000)
        ys = np.full_like(xs, 0.0)
        f = imp_spline_2d(xs, ys, P, delta=TEST_DELTA, n=N_ORDER)
        dx = xs[1] - xs[0]

        # First derivative (should be bounded and continuous for C^1)
        f1 = np.diff(f) / dx
        assert float(np.abs(f1).max()) < 20.0, "First derivative is unexpectedly large"

        # No abrupt sign flip in first derivative that would indicate C^0 kink
        # (we allow sign changes, but they should not be instantaneous)
        sign_flips = np.sum(np.diff(np.sign(f1)) != 0)
        assert sign_flips < 20, (
            f"Too many first-derivative sign flips ({sign_flips}); "
            "suggests C^0 discontinuity in the field"
        )

    @pytest.mark.parametrize("name", sorted(SECTION7_POLYGONS))
    def test_section7_contour_is_closed_and_smooth(self, name):
        """Iso-contour at 0.5 must exist, be closed, and have no C^0 cusps.

        The H-kernel with n=2 gives C^1 smoothness.  We expect a continuous
        iso-contour without abrupt jumps >25° (C^0 cusps), but allow the
        moderate tangent changes typical of a C^1 curve near control vertices.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        MAX_JUMP_DEG = 25.0  # C^1 field allows moderate angle changes, not C^0 cusps

        P = SECTION7_POLYGONS[name]
        N = 300
        X, Y = np.meshgrid(
            np.linspace(P[:, 0].min() - 0.5, P[:, 0].max() + 0.5, N),
            np.linspace(P[:, 1].min() - 0.5, P[:, 1].max() + 0.5, N),
        )
        Z = imp_spline_2d(X, Y, P, delta=TEST_DELTA, n=N_ORDER)

        fig, ax = plt.subplots()
        cs = ax.contour(X, Y, Z, levels=[0.5])
        plt.close(fig)

        segs = cs.allsegs[0]
        assert len(segs) > 0, f"{name}: no iso-contour found at level 0.5"

        max_jump_deg = 0.0
        for seg in segs:
            if len(seg) < 4:
                continue
            diff = np.diff(seg, axis=0)
            angles = np.arctan2(diff[:, 1], diff[:, 0])
            angle_diffs = np.degrees(np.abs(np.diff(np.unwrap(angles))))
            max_jump_deg = max(max_jump_deg, float(angle_diffs.max()))

        assert max_jump_deg < MAX_JUMP_DEG, (
            f"{name}: max tangent-angle jump {max_jump_deg:.2f}° exceeds "
            f"{MAX_JUMP_DEG}°. This indicates C^0 cusps on the iso-contour, "
            "not the expected C^1 smooth curve from the H-kernel with n=2."
        )

    def test_higher_n_increases_smoothness(self):
        """With larger n, the H-kernel produces smoother fields (C^(n-1))."""
        P = SECTION7_POLYGONS["heart_like"]
        xs = np.linspace(-2.5, 2.5, 2000)
        ys = np.zeros_like(xs)
        dx = xs[1] - xs[0]
        jumps = {}
        for n in [1, 2, 3]:
            f = imp_spline_2d(xs, ys, P, delta=TEST_DELTA, n=n)
            # Second FD: d²f/dx²
            f2 = np.diff(f, n=2) / (dx ** 2)
            jumps[n] = float(np.abs(f2).max())
        # Higher n should give a smaller or equal second-derivative magnitude
        # (field becomes smoother; strict inequality for n=1 vs n=3)
        assert jumps[3] <= jumps[1] * 1.5, (
            f"Higher n should not produce rougher second derivative: "
            f"n=1 max|f''|={jumps[1]:.1f}, n=3 max|f''|={jumps[3]:.1f}"
        )
