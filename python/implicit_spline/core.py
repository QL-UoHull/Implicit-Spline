"""
core.py
-------
Core mathematical functions for 2D piecewise algebraic implicit splines.

All public functions are fully vectorised (NumPy array-in, array-out) and
mirror the MATLAB functions in  matlab/  with identical sign conventions and
parameter names.

Polygon evaluation
------------------
``imp_spline_2d`` supports both **convex** and **non-convex** simple polygons:

* **Convex polygons** use the per-edge product construction
  ``f = ∏_i H(L_i, δ, n)`` (exact, unchanged from the original paper).
* **Non-convex polygons** use a signed-distance construction:
  the unsigned distance to the nearest polygon boundary segment is computed,
  then signed by whether the query point is inside or outside the polygon
  (ray-casting), and finally passed through ``H``.  This correctly handles
  reflex vertices and produces the full shape field rather than collapsing to
  the polygon kernel.

For a non-convex polygon ``Ω`` split into non-overlapping convex sub-polygons
``Ω_i``, the bounded smooth union ``convex_decomp_field`` gives a field that
is numerically equivalent to the direct ``imp_spline_2d`` evaluation away from
shared internal decomposition boundaries (within a band of width ``≈ 2δ``
around shared edges, the constructions differ—the direct evaluation gives the
correct non-zero interior value while the per-piece boundary-respecting
construction necessarily gives zero at the shared edges).

Reference
---------
Li, Q. & Tian, J. (2009).  2D Piecewise Algebraic Splines for Implicit
Modeling.  ACM Transactions on Graphics, 28(3).
DOI: 10.1145/1516522.1516524
"""

import warnings
from math import comb

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Smooth step function
# ──────────────────────────────────────────────────────────────────────────────

def H(t, delta: float = 1.0, n: int = 2) -> np.ndarray:
    """Smooth Heaviside-like step function of order *n*.

    Returns a smooth monotone function that transitions from 0 (at ``t <= 0``)
    to 1 (at ``t >= delta``) using a degree-``(2*n+1)`` Bernstein polynomial
    whose first *n* derivatives vanish at both transition points, giving C^n
    continuity.

    Parameters
    ----------
    t : array-like
        Input values (scalar or any shape).
    delta : float, optional
        Transition width.  Must be positive.  Default: 1.0.
    n : int, optional
        Smoothness order.  The result is C^n at ``t=0`` and ``t=delta``.
        ``n=1`` → cubic smoothstep; ``n=2`` → quintic (recommended); etc.
        Default: 2.

    Returns
    -------
    numpy.ndarray
        Values in [0, 1], same shape as *t*.

    Raises
    ------
    ValueError
        If ``delta <= 0`` or ``n < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(-0.2, 1.2, 5)
    >>> H(t, delta=1.0, n=2)
    array([0.   , 0.   , 0.5  , 1.   , 1.   ])
    """
    if delta <= 0:
        raise ValueError(f"H: delta must be positive (got {delta}).")
    if n < 1 or int(n) != n:
        raise ValueError(f"H: n must be a positive integer (got {n}).")
    n = int(n)

    t = np.asarray(t, dtype=float)
    s = np.clip(t / delta, 0.0, 1.0)

    # Bernstein polynomial sum:
    #   H_n(s) = ∑_{k=n+1}^{2n+1}  C(2n+1, k) · s^k · (1-s)^{2n+1-k}
    h = np.zeros_like(s)
    for k in range(n + 1, 2 * n + 2):
        h += comb(2 * n + 1, k) * (s ** k) * ((1.0 - s) ** (2 * n + 1 - k))
    return h


# ──────────────────────────────────────────────────────────────────────────────
# Signed-distance primitives
# ──────────────────────────────────────────────────────────────────────────────

def lxy(x, y,
        x1: float, y1: float,
        x2: float, y2: float) -> np.ndarray:
    """Normalised signed distance from ``(x, y)`` to directed line ``(x1,y1)→(x2,y2)``.

    Points to the **LEFT** of the directed edge receive a **positive** value,
    which coincides with the interior of a counter-clockwise polygon.

    Formula::

        dx = x2 - x1;  dy = y2 - y1;  length = sqrt(dx² + dy²)
        L  = (dx·(y - y1) - dy·(x - x1)) / length

    Parameters
    ----------
    x, y : array-like
        Query point(s).
    x1, y1, x2, y2 : float
        Directed edge endpoints.

    Returns
    -------
    numpy.ndarray
        Signed distances, same shape as *x* / *y*.

    Warns
    -----
    UserWarning
        If the edge has near-zero length; returns zeros in that case.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)

    if length < 1e-12:
        warnings.warn(
            f"lxy: edge ({x1},{y1})->({x2},{y2}) has near-zero length; "
            "returning zeros.",
            UserWarning, stacklevel=2,
        )
        return np.zeros_like(x)

    return (dx * (y - y1) - dy * (x - x1)) / length


def lxy00(x, y,
          x1: float, y1: float,
          x2: float, y2: float) -> np.ndarray:
    """Unnormalised signed linear function for directed line ``(x1,y1)→(x2,y2)``.

    Same sign convention as :func:`lxy` but without dividing by edge length.
    Useful when only the sign matters or when keeping the expression as a
    polynomial factor.

    Parameters
    ----------
    x, y : array-like
        Query point(s).
    x1, y1, x2, y2 : float
        Directed edge endpoints.

    Returns
    -------
    numpy.ndarray
        Unnormalised signed values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x2 - x1
    dy = y2 - y1
    return dx * (y - y1) - dy * (x - x1)


def point_imp(x, y, px: float, py: float) -> np.ndarray:
    """Squared Euclidean distance from ``(x, y)`` to fixed point ``(px, py)``.

    Returns ``(x - px)² + (y - py)²``, a non-negative polynomial that
    can act as a vertex-proximity primitive.

    Parameters
    ----------
    x, y : array-like
        Query point(s).
    px, py : float
        Fixed reference point.

    Returns
    -------
    numpy.ndarray
        Non-negative squared distances.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (x - px) ** 2 + (y - py) ** 2


# ──────────────────────────────────────────────────────────────────────────────
# Polygon utilities
# ──────────────────────────────────────────────────────────────────────────────

def polygon_signed_area(P) -> float:
    """Signed area of polygon *P* (positive = counter-clockwise).

    Uses the Shoelace (Gauss's area) formula.

    Parameters
    ----------
    P : array-like, shape (m, 2)
        Polygon vertices.

    Returns
    -------
    float
        Signed area.  Positive for CCW, negative for CW.
    """
    P = np.asarray(P, dtype=float)
    x, y = P[:, 0], P[:, 1]
    xn = np.roll(x, -1)
    yn = np.roll(y, -1)
    return 0.5 * float(np.sum(x * yn - xn * y))


# ──────────────────────────────────────────────────────────────────────────────
# Private geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unsigned_seg_dist(x, y, x1: float, y1: float,
                       x2: float, y2: float) -> np.ndarray:
    """Unsigned distance from each point ``(x, y)`` to segment ``(x1,y1)→(x2,y2)``.

    Clamps the foot-of-perpendicular parameter *t* to [0, 1], so the result
    is the distance to the nearest point on the closed segment (not the
    infinite line).

    Parameters
    ----------
    x, y : array-like
        Query coordinates (vectorised).
    x1, y1, x2, y2 : float
        Segment endpoints.

    Returns
    -------
    numpy.ndarray
        Non-negative distances, same shape as *x*.
    """
    dx = x2 - x1
    dy = y2 - y1
    # Use squared length to avoid sqrt; 1e-24 guards against squared-length
    # underflow (segment shorter than ~1e-12 in each coordinate).
    len2 = dx * dx + dy * dy
    if len2 < 1e-24:
        return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    t = np.clip(((x - x1) * dx + (y - y1) * dy) / len2, 0.0, 1.0)
    px = x1 + t * dx
    py = y1 + t * dy
    return np.sqrt((x - px) ** 2 + (y - py) ** 2)


def _point_in_polygon(x, y, P) -> np.ndarray:
    """Boolean array: True where ``(x, y)`` is strictly inside polygon *P*.

    Uses the ray-casting (even-odd) rule.  Points exactly on an edge may be
    classified as inside or outside depending on floating-point arithmetic,
    but those points have zero signed distance and their ``H`` value is 0
    regardless of the inside/outside flag.

    Parameters
    ----------
    x, y : numpy.ndarray
        Query coordinates (same shape).
    P : numpy.ndarray, shape (m, 2)
        Polygon vertices (CCW or CW; orientation does not affect the result).

    Returns
    -------
    numpy.ndarray of bool
        Same shape as *x*.
    """
    inside = np.zeros(x.shape, dtype=bool)
    m = len(P)
    j = m - 1
    for i in range(m):
        xi, yi = P[i, 0], P[i, 1]
        xj, yj = P[j, 0], P[j, 1]
        # Ray from (x,y) in the +x direction. Horizontal edges (yi==yj) are
        # skipped because (yi > y) != (yj > y) is False for them.
        crosses_y = (yi > y) != (yj > y)
        if np.any(crosses_y):
            denom = yj - yi
            # Safe division: denom != 0 wherever crosses_y is True.
            x_intersect = np.where(
                crosses_y,
                # 1e-30: strictly below machine epsilon squared; safe divisor
                # when crosses_y is True (denom != 0 by construction).
                (xj - xi) * (y - yi) / np.where(np.abs(denom) < 1e-30, 1.0, denom) + xi,
                0.0,
            )
            inside ^= crosses_y & (x < x_intersect)
        j = i
    return inside


def _segments_properly_intersect(p1, p2, p3, p4) -> bool:
    """True if open segments ``p1-p2`` and ``p3-p4`` properly intersect.

    "Properly" means the crossing point is strictly in the interior of both
    segments (no shared endpoints, no collinear touching).
    """
    r = p2 - p1
    s = p4 - p3
    rs = float(r[0] * s[1] - r[1] * s[0])
    if abs(rs) < 1e-12:
        return False
    qp = p3 - p1
    t = float(qp[0] * s[1] - qp[1] * s[0]) / rs
    u = float(qp[0] * r[1] - qp[1] * r[0]) / rs
    return 0.0 < t < 1.0 and 0.0 < u < 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Public polygon utilities
# ──────────────────────────────────────────────────────────────────────────────

def is_convex(P) -> bool:
    """Return ``True`` if polygon *P* is (weakly) convex.

    Checks that all cross products of consecutive edge vectors have the same
    sign (or are zero, indicating collinear vertices).  Works for both CW and
    CCW orderings.

    Parameters
    ----------
    P : array-like, shape (m, 2)
        Polygon vertices (closed polygon; do not repeat the first vertex).

    Returns
    -------
    bool
        ``True`` if *P* is convex, ``False`` if it has one or more reflex
        vertices.

    Examples
    --------
    >>> import numpy as np
    >>> is_convex(np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float))
    True
    >>> is_convex(np.array([[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]], dtype=float))
    False
    """
    P = np.asarray(P, dtype=float)
    m = len(P)
    sign = 0
    for i in range(m):
        dx1 = P[(i + 1) % m, 0] - P[i, 0]
        dy1 = P[(i + 1) % m, 1] - P[i, 1]
        dx2 = P[(i + 2) % m, 0] - P[(i + 1) % m, 0]
        dy2 = P[(i + 2) % m, 1] - P[(i + 1) % m, 1]
        cross = dx1 * dy2 - dy1 * dx2
        if abs(cross) > 1e-12:
            s = 1 if cross > 0 else -1
            if sign == 0:
                sign = s
            elif s != sign:
                return False
    return True


def polygon_validate(P, *, name: str = "P"):
    """Validate and normalize a simple polygon.

    Performs the following steps in order:

    1. Remove a repeated closing vertex (``P[-1] == P[0]``).
    2. Remove duplicate consecutive vertices.
    3. Raise ``ValueError`` if fewer than 3 unique vertices remain.
    4. Raise ``ValueError`` if any edge has near-zero length.
    5. Raise ``ValueError`` if any non-adjacent edges properly intersect
       (self-intersecting / non-simple polygon).
    6. Normalize orientation to **counter-clockwise** (positive signed area).

    Parameters
    ----------
    P : array-like, shape (m, 2)
        Polygon vertices.  May include a repeated closing vertex.
    name : str, optional
        Name used in error messages.  Default: ``"P"``.

    Returns
    -------
    numpy.ndarray, shape (m', 2)
        Cleaned polygon in CCW order.  ``m' <= m``.

    Raises
    ------
    ValueError
        On any geometric degeneracy or self-intersection.

    Examples
    --------
    >>> import numpy as np
    >>> P_cw = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=float)   # CW square
    >>> Pv = polygon_validate(P_cw)
    >>> polygon_signed_area(Pv) > 0    # normalised to CCW
    True
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(
            f"polygon_validate: {name} must be shape (m, 2), got {P.shape}."
        )

    # 1. Remove repeated closing vertex
    if len(P) >= 2 and np.allclose(P[0], P[-1], atol=1e-12):
        P = P[:-1]

    # 2. Remove duplicate consecutive vertices
    keep = np.ones(len(P), dtype=bool)
    for i in range(1, len(P)):
        if np.allclose(P[i], P[i - 1], atol=1e-12):
            keep[i] = False
    P = P[keep]

    # 3. At least 3 unique vertices
    if len(P) < 3:
        raise ValueError(
            f"polygon_validate: {name} must have at least 3 unique vertices "
            f"(got {len(P)} after removing duplicates)."
        )

    # 4. No zero-length edges
    m = len(P)
    for i in range(m):
        j = (i + 1) % m
        dx = P[j, 0] - P[i, 0]
        dy = P[j, 1] - P[i, 1]
        if np.sqrt(dx * dx + dy * dy) < 1e-12:
            raise ValueError(
                f"polygon_validate: {name} has a zero-length edge between "
                f"vertices {i} and {j}."
            )

    # 5. No self-intersections among non-adjacent edges
    for i in range(m):
        j = (i + 1) % m
        for k in range(i + 2, m):
            ll = (k + 1) % m
            if ll == i:
                continue  # adjacent edge — skip
            if _segments_properly_intersect(P[i], P[j], P[k], P[ll]):
                raise ValueError(
                    f"polygon_validate: {name} is self-intersecting "
                    f"(edges {i}-{j} and {k}-{ll} cross)."
                )

    # 6. Normalize to CCW
    if polygon_signed_area(P) < 0:
        P = P[::-1].copy()

    return P


def triangulate_polygon(P):
    """Triangulate a simple polygon using the ear-clipping algorithm.

    Accepts both convex and concave (non-self-intersecting) polygons.  The
    vertices need not be in any particular order; the function normalizes to
    CCW internally.

    Parameters
    ----------
    P : array-like, shape (m, 2)
        Polygon vertices.

    Returns
    -------
    list of numpy.ndarray, each shape (3, 2)
        Triangle vertex coordinates.  The triangulation has exactly ``m - 2``
        triangles and covers the interior of *P* without gaps or overlaps.

    Raises
    ------
    ValueError
        If *P* has fewer than 3 vertices.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]], dtype=float)
    >>> tris = triangulate_polygon(P)
    >>> len(tris)      # 6 vertices → 4 triangles
    4
    """
    P = np.asarray(P, dtype=float)
    m = len(P)
    if m < 3:
        raise ValueError(
            f"triangulate_polygon: need at least 3 vertices (got {m})."
        )
    if polygon_signed_area(P) < 0:
        P = P[::-1].copy()

    indices = list(range(m))
    triangles = []

    def _is_ear(idx_list, k):
        n = len(idx_list)
        i = idx_list[(k - 1) % n]
        j = idx_list[k]
        ll = idx_list[(k + 1) % n]
        ax, ay = P[i]
        bx, by = P[j]
        cx, cy = P[ll]
        # Must be a convex (left-turn) vertex in the current polygon.
        # cross > 1e-12 excludes reflex vertices (cross < 0) AND near-collinear
        # vertices (cross ≈ 0) which would produce degenerate zero-area triangles.
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if cross <= 1e-12:
            return False
        # No other remaining vertex may lie strictly inside triangle i-j-l
        tri = np.array([[ax, ay], [bx, by], [cx, cy]])
        for idx_other in idx_list:
            if idx_other in (i, j, ll):
                continue
            if _pt_in_tri(P[idx_other], tri):
                return False
        return True

    def _pt_in_tri(p, tri):
        a, b, c = tri
        d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
        d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
        d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    max_iters = m * m + m
    iters = 0
    while len(indices) > 3:
        iters += 1
        if iters > max_iters:
            break
        n = len(indices)
        found = False
        for k in range(n):
            if _is_ear(indices, k):
                i = indices[(k - 1) % n]
                j = indices[k]
                ll = indices[(k + 1) % n]
                triangles.append(P[[i, j, ll]])
                indices.pop(k)
                found = True
                break
        if not found:
            break

    if len(indices) >= 3:
        triangles.append(P[indices[:3]])

    return triangles


# ──────────────────────────────────────────────────────────────────────────────
# Main implicit-spline function
# ──────────────────────────────────────────────────────────────────────────────

def imp_spline_2d(x, y, P, delta: float = 0.1, n: int = 2) -> np.ndarray:
    """2D piecewise algebraic implicit spline for polygon *P*.

    Evaluates the smooth implicit function at query point(s) ``(x, y)``
    using a construction that is correct for **both convex and non-convex**
    simple polygons:

    **Convex polygons** — per-edge product construction (original paper):

    .. math::

        f(x,y) = \\prod_{i=0}^{m-1}  H\\!\\left(L_i(x,y),\\, \\delta,\\, n\\right)

    where :math:`L_i` is the normalised signed distance from ``(x,y)`` to
    the infinite line through edge *i* (positive on the interior side).

    **Non-convex polygons** — signed-distance construction:

    .. math::

        f(x,y) = H\\!\\bigl(d_\\pm(x,y),\\, \\delta,\\, n\\bigr)

    where :math:`d_\\pm` is the distance to the nearest point on any boundary
    segment, positive inside the polygon and negative outside (determined by
    ray-casting).  This correctly handles reflex vertices.

    In both cases the result is:

    * ``f ≈ 1``  deep inside the polygon
    * ``f = 0``  on the polygon boundary
    * ``f ≈ 0``  outside the polygon

    and is C^n continuous near each polygon edge.

    Parameters
    ----------
    x, y : array-like
        Query point(s).  Scalars or arrays of matching shape.
    P : array-like, shape (m, 2)
        Polygon vertices.  May be in CW or CCW order; the function
        auto-corrects to CCW.  The polygon need not be explicitly closed.
    delta : float, optional
        Transition bandwidth.  Recommended: 5–30 % of the polygon's
        characteristic size (e.g. inscribed-circle radius).  Default: 0.1.
    n : int, optional
        Smoothness order.  The result is C^n near each edge.  Default: 2.

    Returns
    -------
    numpy.ndarray
        Implicit function values in [0, 1], same shape as *x* / *y*.

    Raises
    ------
    ValueError
        If *P* has fewer than 3 vertices, or if ``delta <= 0``.

    Notes
    -----
    Convex-polygon behavior is identical to the original implementation so
    that all existing code relying on convex polygons is unaffected.

    For the signed-distance path (non-convex), the field satisfies
    ``B_Ω ≈ Σ_i B_{Ω_i}`` at points more than ``2δ`` from any shared
    internal decomposition boundary; within that band the piece-wise
    construction gives zero at shared edges while the direct evaluation
    gives the correct interior value.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    >>> imp_spline_2d(0.5, 0.5, P, delta=0.1, n=2)
    array(1.)
    >>> imp_spline_2d(0.0, 0.5, P, delta=0.1, n=2)
    array(0.)
    >>> # Non-convex L-shape: interior point near reflex vertex
    >>> L = np.array([[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]], dtype=float)
    >>> float(imp_spline_2d(0.5, 1.0, L, delta=0.05, n=2)) > 0.5
    True

    Reference
    ---------
    Li, Q. & Tian, J. (2009).  2D Piecewise Algebraic Splines for Implicit
    Modeling.  ACM Transactions on Graphics, 28(3).
    DOI: 10.1145/1516522.1516524
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    P = np.asarray(P, dtype=float)

    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(
            f"imp_spline_2d: P must be shape (m, 2), got {P.shape}."
        )
    m = len(P)
    if m < 3:
        raise ValueError(
            f"imp_spline_2d: polygon must have at least 3 vertices (got {m})."
        )
    if delta <= 0:
        raise ValueError(f"imp_spline_2d: delta must be positive (got {delta}).")

    # Ensure counter-clockwise orientation
    if polygon_signed_area(P) < 0:
        P = P[::-1]

    if is_convex(P):
        # ── Convex path: original per-edge product construction ──────────────
        # Exact for convex polygons and backward-compatible with all prior code.
        f = np.ones_like(x)
        for i in range(m):
            j = (i + 1) % m
            L_i = lxy(x, y, P[i, 0], P[i, 1], P[j, 0], P[j, 1])
            f = f * H(L_i, delta, n)
        return f

    # ── Non-convex path: signed-distance construction ────────────────────────
    # 1. Unsigned distance to the nearest boundary segment.
    d_min = np.full(x.shape, np.inf)
    for i in range(m):
        j = (i + 1) % m
        d = _unsigned_seg_dist(x, y, P[i, 0], P[i, 1], P[j, 0], P[j, 1])
        d_min = np.minimum(d_min, d)

    # 2. Sign: positive inside, negative outside (ray-casting).
    inside = _point_in_polygon(x, y, P)
    signed_d = np.where(inside, d_min, -d_min)

    # 3. Smooth step.
    return H(signed_d, delta, n)


# ──────────────────────────────────────────────────────────────────────────────
# Composition helpers for concave polygons and polygon partitions
# ──────────────────────────────────────────────────────────────────────────────

def smooth_union(a, b) -> np.ndarray:
    """Bounded smooth union of two implicit fields.

    Computes ``1 - (1 - a) * (1 - b)``, which satisfies:

    * Result lies in ``[0, 1]`` whenever both inputs lie in ``[0, 1]``.
    * ``smooth_union(0, b) == b`` and ``smooth_union(a, 0) == a``.
    * ``smooth_union(1, b) == 1`` for any *b*.

    This is the standard "product-of-complements" composition used to combine
    implicit fields from a convex decomposition without exceeding 1.

    Parameters
    ----------
    a, b : array-like
        Input implicit fields.  Values should be in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Composed field in ``[0, 1]``.  Same shape as *a* and *b*.

    Examples
    --------
    >>> import numpy as np
    >>> smooth_union(np.array([0.0, 0.5]), np.array([0.5, 0.5]))
    array([0.5  , 0.75 ])
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 1.0 - (1.0 - a) * (1.0 - b)


def convex_decomp_field(x, y, polygons, delta: float = 0.1, n: int = 2) -> np.ndarray:
    """Evaluate a (concave) shape field via a bounded smooth union of convex pieces.

    The shape's implicit field is:

    .. math::

        B(x,y) = 1 - \\prod_{k} \\bigl(1 - B_k(x,y)\\bigr)

    where :math:`B_k = \\text{imp\\_spline\\_2d}(x, y, P_k, \\delta, n)` is the
    implicit spline for convex piece *k*.

    Unlike a raw sum, this bounded smooth union remains in ``[0, 1]`` for any
    combination of piece fields (even overlapping ones), and equals 0 outside all
    pieces and ≈ 1 deep inside any piece.

    Parameters
    ----------
    x, y : array-like
        Query point(s).
    polygons : list of array-like, each shape (m_k, 2)
        Convex polygon pieces of the decomposition.
    delta : float
        Transition bandwidth.  Default: 0.1.
    n : int
        Smoothness order.  Default: 2.

    Returns
    -------
    numpy.ndarray
        Implicit field values in ``[0, 1]``, same shape as *x* / *y*.

    Examples
    --------
    >>> import numpy as np
    >>> P1 = np.array([[0,0],[2,0],[2,1],[0,1]], dtype=float)   # lower rectangle
    >>> P2 = np.array([[0,1],[1,1],[1,2],[0,2]], dtype=float)   # upper square
    >>> # Together they form an L-shape; point inside should be high
    >>> float(convex_decomp_field(0.5, 0.5, [P1, P2], delta=0.05, n=2)) > 0.5
    True
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = np.zeros_like(x)
    for P in polygons:
        B = imp_spline_2d(x, y, P, delta=delta, n=n)
        result = smooth_union(result, B)
    return result


def partition_basis_normalized(
    polygons,
    x,
    y,
    delta: float = 0.1,
    n: int = 2,
    eps: float = 1e-12,
):
    """Evaluate normalized partition-basis functions for a convex-cell partition.

    For a collection of non-overlapping convex polygons tiling a domain, compute
    the raw implicit spline fields and normalize so that they sum to 1 wherever
    the raw sum exceeds *eps*:

    .. math::

        \\hat{B}_k(x,y)
            = \\frac{B_k(x,y)}{\\max\\!\\bigl(\\sum_j B_j(x,y),\\; \\varepsilon\\bigr)}

    By construction, :math:`\\sum_k \\hat{B}_k = 1` at every point where the raw
    denominator exceeds *eps*; the result is finite everywhere (no division by
    zero).

    Parameters
    ----------
    polygons : list of array-like, each shape (m_k, 2)
        Non-overlapping convex cells whose union defines the partition domain.
    x, y : array-like
        Query point(s).
    delta : float
        Transition bandwidth for each cell field.  Default: 0.1.
    n : int
        Smoothness order.  Default: 2.
    eps : float
        Minimum denominator guard against division by zero.  Default: 1e-12.

    Returns
    -------
    basis : list of numpy.ndarray
        Normalized basis value for each cell, in ``[0, 1]``.
    raw_sum : numpy.ndarray
        Denominator (sum of raw fields) before normalization.  Can be
        compared against *eps* to identify points outside the partition domain.

    Notes
    -----
    The partition-of-unity property ``sum_k hat_B_k == 1`` holds exactly (up
    to floating-point rounding) at any point with ``raw_sum > eps``.  Points
    with ``raw_sum <= eps`` are outside the active region of all cells; their
    normalized values equal the individual raw field divided by *eps*, which
    is near zero for points far from the partition domain.

    Examples
    --------
    >>> import numpy as np
    >>> cells = [np.array([[0,0],[1,0],[1,1],[0,1]],dtype=float),
    ...          np.array([[1,0],[2,0],[2,1],[1,1]],dtype=float)]
    >>> basis, raw_sum = partition_basis_normalized(cells, np.array([0.5,1.5]),
    ...                                             np.array([0.5,0.5]), delta=0.1)
    >>> abs(sum(basis)[0] - 1.0) < 1e-10  # sums to 1 at interior point
    True
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    raw = [
        imp_spline_2d(x, y, np.asarray(P, dtype=float), delta=delta, n=n)
        for P in polygons
    ]
    raw_sum = sum(raw)
    denom = np.maximum(raw_sum, eps)
    basis = [b / denom for b in raw]
    return basis, raw_sum
