"""
core.py
-------
Core mathematical functions for 2D piecewise algebraic implicit splines.

The public evaluator ``imp_spline_2d`` uses a boundary-based additive
construction for arbitrary simple polygons.  The field is assembled as a signed
sum of oriented boundary-edge contributions, so internal edges in a convex
partition cancel exactly and decomposition evaluation matches direct evaluation.

Low-level helpers such as ``H`` and ``lxy`` are retained for legacy convex-only
constructions and for compatibility with the MATLAB primitives.
"""

from __future__ import annotations

import warnings
from math import comb, factorial

import numpy as np


_CLOSE_TOL = 1e-12
_DEGENERATE_EDGE_TOL = 1e-10
_INTERSECTION_TOL = 1e-12
_DEFAULT_QUADRATURE_ORDER = 96


# ──────────────────────────────────────────────────────────────────────────────
# Smooth step function
# ──────────────────────────────────────────────────────────────────────────────

def H(t, delta: float = 1.0, n: int = 2) -> np.ndarray:
    """Smooth Heaviside-like step function of order *n*."""
    if delta <= 0:
        raise ValueError(f"H: delta must be positive (got {delta}).")
    if n < 1 or int(n) != n:
        raise ValueError(f"H: n must be a positive integer (got {n}).")
    n = int(n)

    t = np.asarray(t, dtype=float)
    s = np.clip(t / delta, 0.0, 1.0)

    h = np.zeros_like(s)
    for k in range(n + 1, 2 * n + 2):
        h += comb(2 * n + 1, k) * (s ** k) * ((1.0 - s) ** (2 * n + 1 - k))
    return h


# ──────────────────────────────────────────────────────────────────────────────
# Signed-distance primitives
# ──────────────────────────────────────────────────────────────────────────────

def lxy(x, y, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Normalised signed distance from ``(x, y)`` to directed line ``(x1,y1)→(x2,y2)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)

    if length < _CLOSE_TOL:
        warnings.warn(
            f"lxy: edge ({x1},{y1})->({x2},{y2}) has near-zero length; returning zeros.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros_like(x)

    return (dx * (y - y1) - dy * (x - x1)) / length


def lxy00(x, y, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Unnormalised signed linear function for directed line ``(x1,y1)→(x2,y2)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x2 - x1
    dy = y2 - y1
    return dx * (y - y1) - dy * (x - x1)


def point_imp(x, y, px: float, py: float) -> np.ndarray:
    """Squared Euclidean distance from ``(x, y)`` to fixed point ``(px, py)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (x - px) ** 2 + (y - py) ** 2


# ──────────────────────────────────────────────────────────────────────────────
# Polygon utilities
# ──────────────────────────────────────────────────────────────────────────────

def polygon_signed_area(P) -> float:
    """Signed area of polygon *P* (positive = counter-clockwise)."""
    P = np.asarray(P, dtype=float)
    x, y = P[:, 0], P[:, 1]
    xn = np.roll(x, -1)
    yn = np.roll(y, -1)
    return 0.5 * float(np.sum(x * yn - xn * y))


def _cross(a, b, c) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _orient(a, b, c, tol: float = _INTERSECTION_TOL) -> int:
    val = _cross(a, b, c)
    if val > tol:
        return 1
    if val < -tol:
        return -1
    return 0


def _point_on_segment(p, a, b, tol: float = _INTERSECTION_TOL) -> bool:
    if _orient(a, b, p, tol) != 0:
        return False
    return (
        min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol
        and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol
    )


def _segments_intersection_kind(p1, p2, p3, p4, tol: float = _INTERSECTION_TOL) -> str:
    """Classify intersection of closed segments.

    Returns one of: ``none``, ``proper``, ``endpoint``, ``overlap``.
    """
    o1 = _orient(p1, p2, p3, tol)
    o2 = _orient(p1, p2, p4, tol)
    o3 = _orient(p3, p4, p1, tol)
    o4 = _orient(p3, p4, p2, tol)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return "proper"

    endpoint_hits = []
    if o1 == 0 and _point_on_segment(p3, p1, p2, tol):
        endpoint_hits.append(tuple(np.asarray(p3).tolist()))
    if o2 == 0 and _point_on_segment(p4, p1, p2, tol):
        endpoint_hits.append(tuple(np.asarray(p4).tolist()))
    if o3 == 0 and _point_on_segment(p1, p3, p4, tol):
        endpoint_hits.append(tuple(np.asarray(p1).tolist()))
    if o4 == 0 and _point_on_segment(p2, p3, p4, tol):
        endpoint_hits.append(tuple(np.asarray(p2).tolist()))

    if o1 == o2 == o3 == o4 == 0:
        pts = np.array([p1, p2, p3, p4], dtype=float)
        span_x = pts[:, 0].max() - pts[:, 0].min()
        span_y = pts[:, 1].max() - pts[:, 1].min()
        axis = 0 if span_x >= span_y else 1
        a0, a1 = sorted((p1[axis], p2[axis]))
        b0, b1 = sorted((p3[axis], p4[axis]))
        overlap0 = max(a0, b0)
        overlap1 = min(a1, b1)
        if overlap1 < overlap0 - tol:
            return "none"
        if abs(overlap1 - overlap0) <= tol:
            return "endpoint"
        return "overlap"

    if endpoint_hits:
        return "endpoint"
    return "none"


def is_convex(P) -> bool:
    """Return ``True`` if polygon *P* is (weakly) convex."""
    P = np.asarray(P, dtype=float)
    m = len(P)
    sign = 0
    for i in range(m):
        dx1 = P[(i + 1) % m, 0] - P[i, 0]
        dy1 = P[(i + 1) % m, 1] - P[i, 1]
        dx2 = P[(i + 2) % m, 0] - P[(i + 1) % m, 0]
        dy2 = P[(i + 2) % m, 1] - P[(i + 1) % m, 1]
        cross = dx1 * dy2 - dy1 * dx2
        if abs(cross) > _INTERSECTION_TOL:
            s = 1 if cross > 0 else -1
            if sign == 0:
                sign = s
            elif s != sign:
                return False
    return True


def polygon_validate(
    P,
    *,
    name: str = "P",
    closing_tol: float = _CLOSE_TOL,
    duplicate_tol: float = _CLOSE_TOL,
    degenerate_edge_tol: float = _DEGENERATE_EDGE_TOL,
):
    """Validate and normalize a simple polygon.

    Vertices must already be listed in polygon boundary order (CW or CCW).
    A repeated closing vertex is accepted and removed.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"polygon_validate: {name} must be shape (m, 2), got {P.shape}.")

    if len(P) >= 2 and np.allclose(P[0], P[-1], rtol=0.0, atol=closing_tol):
        P = P[:-1]

    keep = np.ones(len(P), dtype=bool)
    for i in range(1, len(P)):
        if np.allclose(P[i], P[i - 1], rtol=0.0, atol=duplicate_tol):
            keep[i] = False
    P = P[keep]

    if len(P) < 3:
        raise ValueError(
            f"polygon_validate: {name} must have at least 3 unique vertices "
            f"(got {len(P)} after removing duplicates)."
        )

    m = len(P)
    for i in range(m):
        j = (i + 1) % m
        if np.hypot(*(P[j] - P[i])) <= degenerate_edge_tol:
            raise ValueError(
                f"polygon_validate: {name} has a near-zero edge between vertices {i} and {j}."
            )

    for i in range(m):
        j = (i + 1) % m
        for k in range(i + 1, m):
            ll = (k + 1) % m
            if i == k or i == ll or j == k or j == ll:
                continue
            kind = _segments_intersection_kind(P[i], P[j], P[k], P[ll])
            if kind == "proper":
                raise ValueError(
                    f"polygon_validate: {name} is self-intersecting "
                    f"(edges {i}-{j} and {k}-{ll} cross)."
                )
            if kind == "overlap":
                raise ValueError(
                    f"polygon_validate: {name} has collinear overlapping edges "
                    f"({i}-{j} and {k}-{ll})."
                )
            if kind == "endpoint":
                raise ValueError(
                    f"polygon_validate: {name} has invalid touching edges "
                    f"({i}-{j} and {k}-{ll})."
                )

    if abs(polygon_signed_area(P)) <= degenerate_edge_tol:
        raise ValueError(f"polygon_validate: {name} has near-zero signed area.")

    if polygon_signed_area(P) < 0:
        P = P[::-1].copy()
    return P


def triangulate_polygon(P):
    """Triangulate a simple polygon in boundary order using ear clipping.

    Parameters
    ----------
    P : array-like, shape (m, 2)
        Polygon vertices in consecutive boundary order (CW or CCW). The
        function normalizes orientation to CCW, but it does not reorder an
        arbitrary point set into a polygon.
    """
    P = polygon_validate(P, name="P")
    m = len(P)
    if m < 3:
        raise ValueError(f"triangulate_polygon: need at least 3 vertices (got {m}).")
    if m == 3:
        return [P.copy()]

    indices = list(range(m))
    triangles = []

    def _pt_in_or_on_tri(p, tri):
        a, b, c = tri
        o1 = _cross(a, b, p)
        o2 = _cross(b, c, p)
        o3 = _cross(c, a, p)
        has_neg = (o1 < -_INTERSECTION_TOL) or (o2 < -_INTERSECTION_TOL) or (o3 < -_INTERSECTION_TOL)
        has_pos = (o1 > _INTERSECTION_TOL) or (o2 > _INTERSECTION_TOL) or (o3 > _INTERSECTION_TOL)
        return not (has_neg and has_pos)

    def _is_ear(idx_list, k):
        nidx = len(idx_list)
        i = idx_list[(k - 1) % nidx]
        j = idx_list[k]
        ll = idx_list[(k + 1) % nidx]
        tri = P[[i, j, ll]]
        if polygon_signed_area(tri) <= _INTERSECTION_TOL:
            return False
        for idx_other in idx_list:
            if idx_other in (i, j, ll):
                continue
            if _pt_in_or_on_tri(P[idx_other], tri):
                return False
        return True

    guard = m * m + m
    for _ in range(guard):
        if len(indices) == 3:
            break
        found = False
        for k in range(len(indices)):
            if _is_ear(indices, k):
                i = indices[(k - 1) % len(indices)]
                j = indices[k]
                ll = indices[(k + 1) % len(indices)]
                triangles.append(P[[i, j, ll]])
                indices.pop(k)
                found = True
                break
        if not found:
            raise ValueError("triangulate_polygon: ear clipping failed to find a valid ear.")

    if len(indices) != 3:
        raise ValueError("triangulate_polygon: ear clipping did not complete.")

    triangles.append(P[indices].copy())
    if len(triangles) != m - 2:
        raise ValueError(
            f"triangulate_polygon: expected {m - 2} triangles, got {len(triangles)}."
        )
    return triangles


# ──────────────────────────────────────────────────────────────────────────────
# Boundary-based spline basis
# ──────────────────────────────────────────────────────────────────────────────

def _gauss_legendre_rule(order: int = _DEFAULT_QUADRATURE_ORDER):
    pts, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (pts + 1.0), 0.5 * weights


def _cardinal_bspline(order: int, u) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    for k in range(order + 1):
        out += ((-1) ** k) * comb(order, k) * np.maximum(u - k, 0.0) ** (order - 1)
    return out / factorial(order - 1)


def _cardinal_bspline_cdf(order: int, u) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    for k in range(order + 1):
        out += ((-1) ** k) * comb(order, k) * np.maximum(u - k, 0.0) ** order
    return out / factorial(order)


def _box_spline_1d(t, delta: float, n: int) -> np.ndarray:
    u = np.asarray(t, dtype=float) / (2.0 * delta) + 0.5 * n
    return _cardinal_bspline(n, u) / (2.0 * delta)


def _box_spline_1d_cdf(t, delta: float, n: int) -> np.ndarray:
    u = np.asarray(t, dtype=float) / (2.0 * delta) + 0.5 * n
    return _cardinal_bspline_cdf(n, u)


def _edge_contribution(x, y, p0, p1, delta: float, n: int, nodes, weights) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dy = y1 - y0
    if abs(dy) <= _CLOSE_TOL:
        return np.zeros(np.broadcast(x, y).shape, dtype=float)

    dx = x1 - x0
    contrib = np.zeros(np.broadcast(x, y).shape, dtype=float)
    for t, w in zip(nodes, weights):
        xe = x0 + t * dx
        ye = y0 + t * dy
        contrib -= w * _box_spline_1d_cdf(x - xe, delta, n) * _box_spline_1d(y - ye, delta, n) * dy
    return contrib


def _polygon_edges(P):
    return [(P[i], P[(i + 1) % len(P)]) for i in range(len(P))]


def _canonical_point(p, tol: float = _CLOSE_TOL):
    scale = 1.0 / tol
    return tuple(int(np.round(coord * scale)) for coord in np.asarray(p, dtype=float))


def _canonical_edge_key(p0, p1, tol: float = _CLOSE_TOL):
    a = _canonical_point(p0, tol)
    b = _canonical_point(p1, tol)
    return (a, b) if a <= b else (b, a)


def cancel_internal_edges(polygons, tol: float = _CLOSE_TOL):
    """Return exterior oriented edges after canceling duplicated internal edges."""
    edge_map = {}
    for poly in polygons:
        P = polygon_validate(poly)
        for p0, p1 in _polygon_edges(P):
            key = _canonical_edge_key(p0, p1, tol)
            entry = edge_map.setdefault(key, [])
            entry.append((np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)))

    exterior = []
    for key, entries in edge_map.items():
        if len(entries) == 1:
            exterior.append(entries[0])
            continue
        if len(entries) == 2:
            a0, a1 = entries[0]
            b0, b1 = entries[1]
            if np.allclose(a0, b1, rtol=0.0, atol=tol) and np.allclose(a1, b0, rtol=0.0, atol=tol):
                continue
        raise ValueError(f"cancel_internal_edges: invalid edge incidence for key {key}.")
    return exterior


def _evaluate_edges(x, y, edges, delta: float, n: int, quadrature_order: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nodes, weights = _gauss_legendre_rule(quadrature_order)
    result = np.zeros(np.broadcast(x, y).shape, dtype=float)
    for p0, p1 in edges:
        result += _edge_contribution(x, y, p0, p1, delta, n, nodes, weights)
    return result


def convex_product_spline_2d(x, y, P, delta: float = 0.1, n: int = 2) -> np.ndarray:
    """Legacy convex-only product construction ``∏ H(L_i)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    P = polygon_validate(P, name="P")
    if delta <= 0:
        raise ValueError(f"convex_product_spline_2d: delta must be positive (got {delta}).")
    if n < 1 or int(n) != n:
        raise ValueError(f"convex_product_spline_2d: n must be a positive integer (got {n}).")

    f = np.ones(np.broadcast(x, y).shape, dtype=float)
    for i in range(len(P)):
        j = (i + 1) % len(P)
        L_i = lxy(x, y, P[i, 0], P[i, 1], P[j, 0], P[j, 1])
        f *= H(L_i, delta, n)
    return f


def imp_spline_2d(x, y, P, delta: float = 0.1, n: int = 2, quadrature_order: int = _DEFAULT_QUADRATURE_ORDER) -> np.ndarray:
    """Boundary-based 2D piecewise algebraic spline basis for a simple polygon.

    The polygon may be convex or non-convex, but its vertices must be listed in
    consecutive boundary order.  A repeated closing vertex is accepted.
    """
    if delta <= 0:
        raise ValueError(f"imp_spline_2d: delta must be positive (got {delta}).")
    if n < 1 or int(n) != n:
        raise ValueError(f"imp_spline_2d: n must be a positive integer (got {n}).")
    if quadrature_order < 2:
        raise ValueError("imp_spline_2d: quadrature_order must be at least 2.")

    P = polygon_validate(P, name="P")
    return _evaluate_edges(x, y, _polygon_edges(P), delta, int(n), quadrature_order)


# ──────────────────────────────────────────────────────────────────────────────
# Composition helpers for concave polygons and polygon partitions
# ──────────────────────────────────────────────────────────────────────────────

def smooth_union(a, b) -> np.ndarray:
    """Bounded smooth union of two implicit fields.

    Retained as a legacy helper; it is not the additive polygon construction.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 1.0 - (1.0 - a) * (1.0 - b)


def convex_decomp_field(x, y, polygons, delta: float = 0.1, n: int = 2, quadrature_order: int = _DEFAULT_QUADRATURE_ORDER) -> np.ndarray:
    """Evaluate a polygon decomposition via additive boundary contributions.

    Internal edges shared by two CCW cells are canceled exactly before the field
    is evaluated, so the result matches direct evaluation on the original outer
    boundary up to floating-point roundoff.
    """
    edges = cancel_internal_edges(polygons)
    return _evaluate_edges(x, y, edges, delta, int(n), quadrature_order)


def partition_basis_fields(polygons, x, y, delta: float = 0.1, n: int = 2, quadrature_order: int = _DEFAULT_QUADRATURE_ORDER):
    """Evaluate additive basis fields for a conforming polygon partition."""
    basis = [imp_spline_2d(x, y, P, delta=delta, n=n, quadrature_order=quadrature_order) for P in polygons]
    return basis, sum(basis)


def partition_basis_normalized(polygons, x, y, delta: float = 0.1, n: int = 2, eps: float = 1e-12, quadrature_order: int = _DEFAULT_QUADRATURE_ORDER):
    """Legacy normalized partition basis.

    The corrected additive construction uses :func:`partition_basis_fields`.
    This function is retained for compatibility with earlier demos/tests.
    """
    warnings.warn(
        "partition_basis_normalized is legacy; prefer partition_basis_fields for the additive construction.",
        DeprecationWarning,
        stacklevel=2,
    )
    basis, raw_sum = partition_basis_fields(polygons, x, y, delta=delta, n=n, quadrature_order=quadrature_order)
    denom = np.maximum(raw_sum, eps)
    return [b / denom for b in basis], raw_sum


def validate_partition(polygons, outer_polygon=None, *, tol: float = _DEGENERATE_EDGE_TOL):
    """Validate a conforming polygon partition.

    Checks CCW orientation, positive area, shared-edge incidence, absence of
    T-junctions, pairwise interior non-overlap, and total-area coverage of the
    optional outer polygon.
    """
    cells = [polygon_validate(P, name=f"cell[{i}]") for i, P in enumerate(polygons)]
    areas = [polygon_signed_area(P) for P in cells]
    if not all(area > 0 for area in areas):
        raise ValueError("validate_partition: every cell must have positive CCW area.")
    if not all(is_convex(P) for P in cells):
        raise ValueError("validate_partition: every cell must be convex.")

    edge_map = {}
    for idx, P in enumerate(cells):
        for p0, p1 in _polygon_edges(P):
            key = _canonical_edge_key(p0, p1)
            edge_map.setdefault(key, []).append((idx, np.asarray(p0), np.asarray(p1)))

    for key, entries in edge_map.items():
        if len(entries) > 2:
            raise ValueError(f"validate_partition: edge {key} has incidence {len(entries)} > 2.")
        if len(entries) == 2:
            (_, a0, a1), (_, b0, b1) = entries
            if not (
                np.allclose(a0, b1, rtol=0.0, atol=tol)
                and np.allclose(a1, b0, rtol=0.0, atol=tol)
            ):
                raise ValueError(f"validate_partition: shared edge {key} does not appear with opposite orientation.")

    for i, Pi in enumerate(cells):
        for j in range(i + 1, len(cells)):
            Pj = cells[j]
            for a0, a1 in _polygon_edges(Pi):
                for b0, b1 in _polygon_edges(Pj):
                    kind = _segments_intersection_kind(a0, a1, b0, b1)
                    if kind == "proper":
                        raise ValueError(
                            f"validate_partition: cells {i} and {j} have invalid proper edge intersection."
                        )
                    if kind == "overlap":
                        same_edge = (
                            np.allclose(a0, b1, rtol=0.0, atol=tol)
                            and np.allclose(a1, b0, rtol=0.0, atol=tol)
                        ) or (
                            np.allclose(a0, b0, rtol=0.0, atol=tol)
                            and np.allclose(a1, b1, rtol=0.0, atol=tol)
                        )
                        if not same_edge:
                            raise ValueError(
                                f"validate_partition: cells {i} and {j} have partially overlapping edges."
                            )
            centroid_i = Pi.mean(axis=0)
            centroid_j = Pj.mean(axis=0)
            if _point_in_polygon_mask(np.array([centroid_i[0]]), np.array([centroid_i[1]]), Pj)[0]:
                raise ValueError(f"validate_partition: cell {i} lies inside cell {j}.")
            if _point_in_polygon_mask(np.array([centroid_j[0]]), np.array([centroid_j[1]]), Pi)[0]:
                raise ValueError(f"validate_partition: cell {j} lies inside cell {i}.")

    all_vertices = [tuple(v.tolist()) for P in cells for v in P]
    for i, Pi in enumerate(cells):
        for a0, a1 in _polygon_edges(Pi):
            for v in all_vertices:
                vv = np.array(v, dtype=float)
                if np.allclose(vv, a0, rtol=0.0, atol=tol) or np.allclose(vv, a1, rtol=0.0, atol=tol):
                    continue
                if _point_on_segment(vv, a0, a1, tol):
                    raise ValueError("validate_partition: detected a T-junction.")

    total_area = float(sum(areas))
    summary = {
        "cells": cells,
        "areas": areas,
        "total_area": total_area,
        "edge_incidence": {key: len(entries) for key, entries in edge_map.items()},
    }

    if outer_polygon is not None:
        outer = polygon_validate(outer_polygon, name="outer_polygon")
        outer_area = polygon_signed_area(outer)
        if not np.isclose(total_area, outer_area, rtol=0.0, atol=tol):
            raise ValueError(
                f"validate_partition: total cell area {total_area} does not match outer area {outer_area}."
            )
        outer_edges = {_canonical_edge_key(a, b) for a, b in _polygon_edges(outer)}
        boundary_edges = {key for key, entries in edge_map.items() if len(entries) == 1}
        if boundary_edges != outer_edges:
            raise ValueError("validate_partition: outer-boundary edge set does not match the declared outer polygon.")
        summary["outer_polygon"] = outer
        summary["outer_area"] = outer_area

    return summary


def _point_in_polygon_mask(x, y, P) -> np.ndarray:
    inside = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    px = np.asarray(P[:, 0], dtype=float)
    py = np.asarray(P[:, 1], dtype=float)
    j = len(P) - 1
    for i in range(len(P)):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        crosses_y = (yi > y) != (yj > y)
        if np.any(crosses_y):
            denom = yj - yi
            x_intersect = np.where(crosses_y, (xj - xi) * (y - yi) / denom + xi, 0.0)
            inside ^= crosses_y & (x < x_intersect)
        j = i
    return inside
