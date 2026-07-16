import os
import sys

import numpy as np
import pytest

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.join(_tests_dir, '..', 'python')
if os.path.isdir(_python_dir):
    sys.path.insert(0, os.path.abspath(_python_dir))

from implicit_spline.core import (
    convex_decomp_field,
    convex_product_spline_2d,
    imp_spline_2d,
    is_convex,
    partition_basis_fields,
    polygon_signed_area,
    polygon_validate,
    triangulate_polygon,
    validate_partition,
    _segments_intersection_kind,
)
from implicit_spline.paper_examples import (
    PARTITION_CELLS,
    PARTITION_OUTER,
    SECTION7_POLYGONS,
    SECTION7_TEST_POINTS,
)

DELTA = 0.22
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
        assert float(imp_spline_2d(x, y, P, delta=DELTA, n=N_ORDER)) > 0.95
    for x, y in SECTION7_TEST_POINTS[name]["outside"]:
        assert float(imp_spline_2d(x, y, P, delta=DELTA, n=N_ORDER)) < 0.15


@pytest.mark.parametrize("name", sorted(SECTION7_POLYGONS))
def test_direct_equals_triangulation_decomposition(name):
    P = SECTION7_POLYGONS[name]
    tris = triangulate_polygon(P)
    X, Y = np.meshgrid(
        np.linspace(P[:, 0].min() - 0.4, P[:, 0].max() + 0.4, 90),
        np.linspace(P[:, 1].min() - 0.4, P[:, 1].max() + 0.4, 90),
    )
    direct = imp_spline_2d(X, Y, P, delta=DELTA, n=N_ORDER)
    decomp = convex_decomp_field(X, Y, tris, delta=DELTA, n=N_ORDER)
    np.testing.assert_allclose(direct, decomp, rtol=0.0, atol=ABS_TOL)


@pytest.mark.parametrize("name", sorted(SECTION7_POLYGONS))
def test_direct_equals_multiple_valid_decompositions(name):
    P = SECTION7_POLYGONS[name]
    X, Y = np.meshgrid(
        np.linspace(P[:, 0].min() - 0.4, P[:, 0].max() + 0.4, 70),
        np.linspace(P[:, 1].min() - 0.4, P[:, 1].max() + 0.4, 70),
    )
    direct = imp_spline_2d(X, Y, P, delta=DELTA, n=N_ORDER)
    decomp_a = convex_decomp_field(X, Y, triangulate_polygon(P), delta=DELTA, n=N_ORDER)
    decomp_b = convex_decomp_field(X, Y, triangulate_polygon(np.roll(P, -3, axis=0)), delta=DELTA, n=N_ORDER)
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
    direct = imp_spline_2d(X, Y, P, delta=DELTA, n=N_ORDER)
    decomp = convex_decomp_field(X, Y, tris, delta=DELTA, n=N_ORDER)
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
    X, Y = np.meshgrid(np.linspace(1.0, 3.1, 60), np.linspace(1.1, 3.0, 60))
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


def test_polygon_validate_uses_absolute_closing_tolerance_only():
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

    touching = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.0, 2.0], [1.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match="invalid touching edges"):
        polygon_validate(touching)


def test_imp_spline_handles_repeated_closing_vertex():
    P = SECTION7_POLYGONS["heart_like"]
    closed = np.vstack([P, P[0]])
    X, Y = np.meshgrid(np.linspace(-2.8, 2.8, 40), np.linspace(-2.4, 1.5, 40))
    open_field = imp_spline_2d(X, Y, P, delta=DELTA, n=N_ORDER)
    closed_field = imp_spline_2d(X, Y, closed, delta=DELTA, n=N_ORDER)
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
    def setup_method(self):
        import matplotlib
        matplotlib.use('Agg')

    def test_safe_contour_and_draw_imp_spline(self):
        import matplotlib.pyplot as plt
        from implicit_spline.visualization import _safe_contour, draw_imp_spline

        fig, ax = plt.subplots()
        X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        Z = np.zeros_like(X)
        _safe_contour(ax, X, Y, Z, levels=[1.5], colors='k')
        plt.close(fig)

        fig2, ax2 = plt.subplots()
        draw_imp_spline(SECTION7_POLYGONS['heart_like'], delta=DELTA, n=N_ORDER, N=40, ax=ax2, iso_level=1.5)
        plt.close(fig2)
