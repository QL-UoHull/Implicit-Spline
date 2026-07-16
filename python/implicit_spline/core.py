"""
core.py
-------
Core mathematical functions for 2D piecewise algebraic implicit splines.

This module is a faithful Python port of the MATLAB reference implementation
by Li & Tian (2009). The public evaluator ``imp_spline_2d`` is based on the
H-kernel boundary integral, which is derived from the MATLAB ``H.m`` smooth
step function via Green's theorem. This construction:

  1. Uses only the MATLAB ``H`` primitive and its derivative – no Gaussian kernel
     or SciPy dependency.
  2. Matches the MATLAB ``H`` parameter semantics:
       * ``delta`` – transition bandwidth (same role as in MATLAB ``H.m``)
       * ``n``     – smoothness order: field is C^(n-1) near each polygon edge
  3. Handles arbitrary simple polygons (convex and non-convex) without
     signed-distance substitution or kernel/intersection collapse.
  4. Is exactly additive: internal edges in any triangulation or partition
     cancel exactly (to floating-point precision) via Green's theorem, so
     ``convex_decomp_field`` matches direct ``imp_spline_2d`` evaluation.

Construction
------------
The 2D kernel is separable::

    K(Δx, Δy) = K_x(Δx) · K_y(Δy)

where K_x and K_y are derived from the MATLAB H function::

    K_x(t) = H'(t + δ/2, δ, n)    # derivative of H, centred so peak is at t=0
    K_y(t) = H'(t + δ/2, δ, n)    # same in y direction

The CDF-like antiderivative in x is::

    CDF_x(t) = H(t + δ/2, δ, n)   # = 0.5 at t=0 (on the polygon boundary)

By Green's theorem the area integral reduces to a 1D boundary integral:

    f(x, y) = ∬_Ω K_x(x−x′) K_y(y−y′) dx′dy′
            = ∮_{∂Ω, CCW} (1 − CDF_x(x−x′)) · K_y(y−y′) dy′

which is evaluated per edge via Gauss–Legendre quadrature.

MATLAB-to-Python function mapping
----------------------------------
MATLAB function          Python equivalent
--------------------     ----------------------------------
H(t, delta, n)           H(t, delta, n)
Lxy(...)                 lxy(...)
Lxy00(...)               lxy00(...)
Point_imp(...)           point_imp(...)
LineSeg_imp(...)         line_seg_imp(...)
L_corner_inter(...)      l_corner_inter(...)
U_Angle_inter(...)       u_angle_inter(...)
Square_Angle_inter(...)  square_angle_inter(...)
ImpSpline2D(...)         convex_product_spline_2d(...)   [convex-only product formula]
(boundary integral)      imp_spline_2d(...)              [public evaluator, arbitrary polygon]

Reference
---------
Li, Q. & Tian, J. (2009). 2D Piecewise Algebraic Splines for Implicit
Modeling. ACM Transactions on Graphics, 28(3).
DOI: 10.1145/1516522.1516524
"""

from __future__ import annotations

import warnings
from math import comb

import numpy as np


# Tolerances used throughout the geometry layer.
# _CLOSE_TOL           : closing-vertex / exact-coordinate comparisons
# _DEGENERATE_EDGE_TOL : minimum admissible edge length / area
# _INTERSECTION_TOL    : orientation / intersection predicates
_CLOSE_TOL = 1e-12
_DEGENERATE_EDGE_TOL = 1e-10
_INTERSECTION_TOL = 1e-12

# Default Gauss–Legendre quadrature order for the H-kernel boundary integral.
# The integrand is a polynomial of degree O(n) in the Gauss nodes, so 16
# points integrate the relevant terms exactly for n ≤ 6.
_DEFAULT_QUADRATURE_ORDER = 16


# ──────────────────────────────────────────────────────────────────────────────
# MATLAB primitive: H (smooth Heaviside step)
# Source: matlab/H.m
# ──────────────────────────────────────────────────────────────────────────────

def H(t, delta: float = 1.0, n: int = 2) -> np.ndarray:
    """Smooth Heaviside-like step function of order *n*.

    Direct port of MATLAB ``H.m``.  Returns a smooth monotone function
    that transitions from 0 (at ``t ≤ 0``) to 1 (at ``t ≥ delta``) using a
    degree-(2*n+1) Bernstein polynomial satisfying H^(k)(0) = H^(k)(delta) = 0
    for k = 1 … n.  The result is C^n continuous at both transition points.

    Parameters
    ----------
    t : array-like
        Input value(s).
    delta : float
        Transition width; must be positive.  (MATLAB default: 1)
    n : int
        Smoothness order; result is C^n at both ends.  (MATLAB default: 2)

    Returns
    -------
    ndarray
        Values in [0, 1], same shape as *t*.
    """
    if delta <= 0:
        raise ValueError(f"H: delta must be positive (got {delta}).")
    if n < 1 or int(n) != n:
        raise ValueError(f"H: n must be a positive integer (got {n}).")
    n = int(n)

    t = np.asarray(t, dtype=float)
    s = np.clip(t / delta, 0.0, 1.0)  # normalised variable in [0, 1]

    # Bernstein-polynomial sum (matches MATLAB H.m line-for-line)
    h = np.zeros_like(s)
    N = 2 * n + 1
    for k in range(n + 1, N + 1):
        h += comb(N, k) * (s ** k) * ((1.0 - s) ** (N - k))
    return h


def _H_prime(t, delta: float, n: int) -> np.ndarray:
    """Derivative of H with respect to *t*.

    Computes dH/dt for the smooth Heaviside defined in ``H()``.  The result is
    C^(n-1) smooth and is supported on the open interval ``(0, delta)``; it
    equals zero outside that range.

    This is the kernel used in the H-kernel boundary integral.
    """
    n = int(n)
    t = np.asarray(t, dtype=float)
    s = np.clip(t / delta, 0.0, 1.0)

    N = 2 * n + 1
    dh_ds = np.zeros_like(s)
    for k in range(n + 1, N + 1):
        c = comb(N, k)
        if k > 0:
            dh_ds += c * k * (s ** (k - 1)) * ((1.0 - s) ** (N - k))
        if N - k > 0:
            dh_ds -= c * (N - k) * (s ** k) * ((1.0 - s) ** (N - k - 1))

    # Chain rule: dH/dt = (dH/ds) / delta.  Zero outside the transition.
    active = (t > 0.0) & (t < delta)
    return np.where(active, dh_ds / delta, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# MATLAB primitives: Lxy, Lxy00, Point_imp
# Sources: matlab/Lxy.m, matlab/Lxy00.m, matlab/Point_imp.m
# ──────────────────────────────────────────────────────────────────────────────

def lxy(x, y, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Normalised signed distance from ``(x, y)`` to directed line ``(x1,y1)→(x2,y2)``.

    Direct port of MATLAB ``Lxy.m``.  Returns the perpendicular signed
    distance; positive to the LEFT of the directed edge.  For a CCW polygon,
    interior points yield L > 0 for every edge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)
    if length < _CLOSE_TOL:
        warnings.warn(
            f"lxy: edge ({x1},{y1})->({x2},{y2}) has near-zero length; returning zeros.",
            UserWarning, stacklevel=2,
        )
        return np.zeros_like(x)
    return (dx * (y - y1) - dy * (x - x1)) / length


def lxy00(x, y, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Unnormalised signed linear function for directed line ``(x1,y1)→(x2,y2)``.

    Direct port of MATLAB ``Lxy00.m``.  Same sign convention as ``lxy`` but
    without dividing by edge length.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x2 - x1
    dy = y2 - y1
    return dx * (y - y1) - dy * (x - x1)


def point_imp(x, y, px: float, py: float) -> np.ndarray:
    """Squared Euclidean distance from ``(x, y)`` to fixed point ``(px, py)``.

    Direct port of MATLAB ``Point_imp.m``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (x - px) ** 2 + (y - py) ** 2


# ──────────────────────────────────────────────────────────────────────────────
# MATLAB primitives: LineSeg_imp, L_corner_inter, U_Angle_inter, Square_Angle_inter
# Sources: matlab/LineSeg_imp.m, matlab/L_corner_inter.m,
#          matlab/U_Angle_inter.m, matlab/Square_Angle_inter.m
# ──────────────────────────────────────────────────────────────────────────────

def line_seg_imp(x, y, x1: float, y1: float, x2: float, y2: float,
                 delta: float = 0.1, n: int = 2) -> np.ndarray:
    """Smooth implicit function for a finite directed line segment.

    Direct port of MATLAB ``LineSeg_imp.m``.  Returns:
      - positive on the LEFT of the segment body (interior side),
      - zero on the segment line,
      - tapers smoothly to zero beyond both endpoints via a tangential gate.

    Algorithm: ``Lperp * H(t_proj, delta, n) * H(len - t_proj, delta, n)``
    where ``Lperp`` is the perpendicular signed distance and ``t_proj`` is the
    along-segment projection.
    """
    if delta <= 0:
        raise ValueError(f"line_seg_imp: delta must be positive (got {delta}).")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx ** 2 + dy ** 2)

    if length < _CLOSE_TOL:
        # Degenerate: treat as a point, return negative proximity (MATLAB fallback)
        return -point_imp(x, y, x1, y1)

    # Perpendicular signed distance to the supporting line (positive = left)
    Lperp = (dx * (y - y1) - dy * (x - x1)) / length

    # Projection of query point onto the segment axis, in [0, length]
    t_proj = (dx * (x - x1) + dy * (y - y1)) / length

    # Smooth tangential gate: 1 in segment interior, 0 at endpoints
    gate = H(t_proj, delta, n) * H(length - t_proj, delta, n)

    return Lperp * gate


def l_corner_inter(x, y, xA: float, yA: float, xB: float, yB: float,
                   xC: float, yC: float, delta: float = 0.1, n: int = 2) -> np.ndarray:
    """Smooth corner function for a general (convex or acute) polygon vertex.

    Direct port of MATLAB ``L_corner_inter.m``.  Computes::

        f = H(L_AB, delta, n) * H(L_BC, delta, n)

    where L_AB and L_BC are the signed distances to the two edge lines.
    This product is 1 deep inside both half-planes (AND logic) and C^n smooth.
    Use for convex vertices (interior angle < 90° to ~180°).

    Parameters
    ----------
    x, y             : query point(s)
    xA, yA           : tail of the incoming edge (vertex A)
    xB, yB           : the corner vertex (vertex B)
    xC, yC           : head of the outgoing edge (vertex C)
    delta, n         : bandwidth and smoothness order for H
    """
    if delta <= 0:
        raise ValueError(f"l_corner_inter: delta must be positive (got {delta}).")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    L1 = lxy(x, y, xA, yA, xB, yB)   # signed dist to line A→B
    L2 = lxy(x, y, xB, yB, xC, yC)   # signed dist to line B→C
    return H(L1, delta, n) * H(L2, delta, n)


def u_angle_inter(x, y, xA: float, yA: float, xB: float, yB: float,
                  xC: float, yC: float, delta: float = 0.1, n: int = 2) -> np.ndarray:
    """Smooth corner function for an obtuse or reflex polygon vertex.

    Direct port of MATLAB ``U_Angle_inter.m``.  Uses the probabilistic-OR
    (inclusive-OR) formula::

        f = h1 + h2 − h1 * h2

    which is always ≥ max(h1, h2).  Use this at vertices where the interior
    angle > 90° or approaches 180°, so the AND product does not "pinch" the
    field in the wide-angle interior region.

    Parameters
    ----------
    x, y             : query point(s)
    xA, yA           : tail of the incoming edge (vertex A)
    xB, yB           : the corner vertex (vertex B)
    xC, yC           : head of the outgoing edge (vertex C)
    delta, n         : bandwidth and smoothness order for H
    """
    if delta <= 0:
        raise ValueError(f"u_angle_inter: delta must be positive (got {delta}).")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    L1 = lxy(x, y, xA, yA, xB, yB)
    L2 = lxy(x, y, xB, yB, xC, yC)
    h1 = H(L1, delta, n)
    h2 = H(L2, delta, n)
    return h1 + h2 - h1 * h2  # inclusive-OR / probabilistic-OR


def square_angle_inter(x, y, xA: float, yA: float, xB: float, yB: float,
                       xC: float, yC: float, delta: float = 0.1, n: int = 2) -> np.ndarray:
    """Smooth corner function for a right-angle (90°) polygon vertex.

    Direct port of MATLAB ``Square_Angle_inter.m``.  Augments the edge product
    with a radial blend near the corner vertex B to prevent iso-level pinching
    at right angles::

        w            = H(L1, delta, n) * H(L2, delta, n)
        corner_fade  = 1 − H(r_sq, (2*delta)^2, n)
        f            = w * (1 − corner_fade) + corner_fade

    Parameters
    ----------
    x, y               : query point(s)
    xA, yA             : tail of the incoming edge (vertex A)
    xB, yB             : the corner vertex (vertex B)
    xC, yC             : head of the outgoing edge (vertex C)
    delta, n           : bandwidth and smoothness order for H
    """
    if delta <= 0:
        raise ValueError(f"square_angle_inter: delta must be positive (got {delta}).")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    L1 = lxy(x, y, xA, yA, xB, yB)
    L2 = lxy(x, y, xB, yB, xC, yC)
    h1 = H(L1, delta, n)
    h2 = H(L2, delta, n)
    w = h1 * h2
    r_sq = point_imp(x, y, xB, yB)   # squared distance to corner vertex B
    corner_fade = 1.0 - H(r_sq, (2.0 * delta) ** 2, n)  # radial fade near B
    return w * (1.0 - corner_fade) + corner_fade


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
# H-kernel boundary integral (public evaluator)
# ──────────────────────────────────────────────────────────────────────────────

def _gauss_legendre_rule(order: int = _DEFAULT_QUADRATURE_ORDER):
    """Return quadrature nodes in [0,1] and corresponding weights."""
    pts, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (pts + 1.0), 0.5 * weights


def _edge_contribution_H_kernel(x, y, p0, p1, delta: float, n: int, nodes, weights) -> np.ndarray:
    """Evaluate one oriented boundary-edge contribution via the H-kernel.

    Implements the formula (from Green's theorem with separable H-kernel)::

        ∫_0^1 (1 − H(Δx + δ/2, δ, n)) · H'(Δy + δ/2, δ, n) · (y1 − y0) dt

    where Δx = x − x_edge(t), Δy = y − y_edge(t).

    Parameters
    ----------
    delta : float
        Transition bandwidth; maps directly to the MATLAB ``H`` parameter.
    n : int
        Smoothness order; maps directly to the MATLAB ``H`` parameter.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dy_edge = y1 - y0
    if abs(dy_edge) <= _CLOSE_TOL:
        return np.zeros(np.broadcast(x, y).shape, dtype=float)

    dx_edge = x1 - x0
    half_delta = 0.5 * delta
    contrib = np.zeros(np.broadcast(x, y).shape, dtype=float)
    for t_node, w in zip(nodes, weights):
        xe = x0 + t_node * dx_edge
        ye = y0 + t_node * dy_edge
        Dx = x - xe      # Δx = x − x_edge
        Dy = y - ye      # Δy = y − y_edge
        # CDF_x: centred H, = 0.5 at Δx = 0, 0 at Δx = −δ/2, 1 at Δx = δ/2
        cdf_x = H(Dx + half_delta, delta, n)
        # K_y: derivative of centred H, peak at Δy = 0, support [−δ/2, δ/2]
        k_y = _H_prime(Dy + half_delta, delta, n)
        contrib += w * (1.0 - cdf_x) * k_y * dy_edge
    return contrib


def _polygon_edges(P) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the consecutive directed edges of polygon ``P``."""
    return [(P[i], P[(i + 1) % len(P)]) for i in range(len(P))]


def _canonical_point(p, tol: float = _CLOSE_TOL):
    """Map a floating-point coordinate to an integer hash key at tolerance ``tol``."""
    scale = 1.0 / tol
    return tuple(int(np.round(coord * scale)) for coord in np.asarray(p, dtype=float))


def _canonical_edge_key(p0, p1, tol: float = _CLOSE_TOL):
    """Create an orientation-independent edge identifier for hashing/matching."""
    a = _canonical_point(p0, tol)
    b = _canonical_point(p1, tol)
    return (a, b) if a <= b else (b, a)


def cancel_internal_edges(polygons, tol: float = _CLOSE_TOL):
    """Return exterior oriented edges after canceling duplicated internal edges.

    For a valid conforming partition (or triangulation), every internal edge
    appears exactly twice in opposite orientations and cancels.  The returned
    list contains only the external boundary edges.
    """
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
    """Evaluate the H-kernel boundary integral for a list of oriented edges."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nodes, weights = _gauss_legendre_rule(quadrature_order)
    result = np.zeros(np.broadcast(x, y).shape, dtype=float)
    for p0, p1 in edges:
        result += _edge_contribution_H_kernel(x, y, p0, p1, delta, n, nodes, weights)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public evaluators
# ──────────────────────────────────────────────────────────────────────────────

def convex_product_spline_2d(x, y, P, delta: float = 0.1, n: int = 2) -> np.ndarray:
    """Product-formula implicit spline: ∏ H(L_i, delta, n).

    This is a direct Python port of MATLAB ``ImpSpline2D.m``.  The function
    is C^n smooth and equals 1 deep inside and 0 at the boundary for convex
    polygons.  For non-convex polygons the value may be reduced near reflex
    vertices; use ``imp_spline_2d`` for arbitrary simple polygons.

    Parameters
    ----------
    x, y  : query point(s)
    P     : (m, 2) array of polygon vertices (CW or CCW; auto-corrected to CCW)
    delta : transition bandwidth (MATLAB default 0.1)
    n     : smoothness order; result is C^n near each edge (MATLAB default 2)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    P = polygon_validate(P, name="P")
    if delta <= 0:
        raise ValueError(f"convex_product_spline_2d: delta must be positive (got {delta}).")
    if n < 1 or int(n) != n:
        raise ValueError(f"convex_product_spline_2d: n must be a positive integer (got {n}).")

    # Product of H(L_i) for each edge — matches MATLAB ImpSpline2D.m loop
    f = np.ones(np.broadcast(x, y).shape, dtype=float)
    for i in range(len(P)):
        j = (i + 1) % len(P)
        L_i = lxy(x, y, P[i, 0], P[i, 1], P[j, 0], P[j, 1])
        f *= H(L_i, delta, n)
    return f


def imp_spline_2d(
    x, y, P, delta: float = 0.1, n: int = 2,
    quadrature_order: int = _DEFAULT_QUADRATURE_ORDER,
) -> np.ndarray:
    """H-kernel boundary-integral implicit spline for an arbitrary simple polygon.

    Public evaluator for arbitrary (convex or non-convex) simple polygons.
    Uses a boundary integral with the smooth H-derivative kernel derived from
    the MATLAB ``H.m`` primitive — no Gaussian kernel or SciPy dependency.

    Properties
    ----------
    * **Non-convex support**: handles any simple polygon without signed-distance
      substitution or interior-region collapse near reflex vertices.
    * **Additivity**: internal edges in any triangulation/partition cancel
      exactly, so ``convex_decomp_field`` matches direct evaluation to
      floating-point precision.
    * **Smoothness**: the field is C^(n-1) near each polygon edge.
    * **MATLAB parity**: ``delta`` and ``n`` carry the same meanings as in
      MATLAB ``H.m`` — larger delta widens the transition zone, larger n
      increases smoothness.

    Parameters
    ----------
    x, y  : query point(s)
    P     : (m, 2) array of polygon vertices (CW or CCW; auto-corrected to CCW)
    delta : transition bandwidth (same role as in MATLAB ``H.m``)
    n     : smoothness order (same role as in MATLAB ``H.m``)
    quadrature_order : Gauss–Legendre points per edge (default 16)
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
# Composition helpers for non-convex polygons and polygon partitions
# ──────────────────────────────────────────────────────────────────────────────

def smooth_union(a, b) -> np.ndarray:
    """Bounded smooth union of two implicit fields: ``1 − (1−a)(1−b)``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 1.0 - (1.0 - a) * (1.0 - b)


def convex_decomp_field(
    x, y, polygons, delta: float = 0.1, n: int = 2,
    quadrature_order: int = _DEFAULT_QUADRATURE_ORDER,
) -> np.ndarray:
    """Evaluate a polygon decomposition via additive H-kernel boundary contributions.

    Internal edges shared by two CCW cells cancel exactly (by Green's theorem)
    before the field is evaluated, so the result matches direct evaluation of
    the outer polygon via ``imp_spline_2d`` to floating-point precision.
    """
    edges = cancel_internal_edges(polygons)
    return _evaluate_edges(x, y, edges, delta, int(n), quadrature_order)


def partition_basis_fields(
    polygons, x, y, delta: float = 0.1, n: int = 2,
    quadrature_order: int = _DEFAULT_QUADRATURE_ORDER,
):
    """Evaluate additive H-kernel basis fields for a conforming polygon partition.

    Returns
    -------
    basis : list of ndarray
        Per-cell field values.
    total : ndarray
        Sum of all basis fields (= ``imp_spline_2d`` of the outer polygon).
    """
    basis = [
        imp_spline_2d(x, y, P, delta=delta, n=n, quadrature_order=quadrature_order)
        for P in polygons
    ]
    return basis, sum(basis)


def partition_basis_normalized(
    polygons, x, y, delta: float = 0.1, n: int = 2, eps: float = 1e-12,
    quadrature_order: int = _DEFAULT_QUADRATURE_ORDER,
):
    """Legacy normalized partition basis (deprecated).

    The corrected additive construction uses :func:`partition_basis_fields`.
    This function is retained for compatibility with earlier demos/tests.
    """
    warnings.warn(
        "partition_basis_normalized is legacy; prefer partition_basis_fields for the additive construction.",
        DeprecationWarning,
        stacklevel=2,
    )
    basis, raw_sum = partition_basis_fields(
        polygons, x, y, delta=delta, n=n, quadrature_order=quadrature_order
    )
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
                raise ValueError(
                    f"validate_partition: shared edge {key} does not appear with opposite orientation."
                )

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
            inside_i = _point_in_polygon_mask(Pi[:, 0], Pi[:, 1], Pj)
            inside_j = _point_in_polygon_mask(Pj[:, 0], Pj[:, 1], Pi)
            strict_i = inside_i & ~_point_on_polygon_boundary(Pi, Pj, tol)
            strict_j = inside_j & ~_point_on_polygon_boundary(Pj, Pi, tol)
            if np.any(strict_i):
                raise ValueError(f"validate_partition: a vertex of cell {i} lies inside cell {j}.")
            if np.any(strict_j):
                raise ValueError(f"validate_partition: a vertex of cell {j} lies inside cell {i}.")

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
            raise ValueError(
                "validate_partition: outer-boundary edge set does not match the declared outer polygon."
            )
        summary["outer_polygon"] = outer
        summary["outer_area"] = outer_area

    return summary


def _point_on_polygon_boundary(points, P, tol: float = _INTERSECTION_TOL) -> np.ndarray:
    """Boolean mask for points lying on any edge of polygon ``P``."""
    points = np.asarray(points, dtype=float)
    mask = np.zeros(len(points), dtype=bool)
    for a, b in _polygon_edges(P):
        mask |= np.array([_point_on_segment(pt, a, b, tol) for pt in points], dtype=bool)
    return mask


def _point_in_polygon_mask(x, y, P) -> np.ndarray:
    """Vectorized even-odd ray-casting test for points inside polygon ``P``."""
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
