"""
core.py
-------
Core mathematical functions for 2D piecewise algebraic implicit splines.

All public functions are fully vectorised (NumPy array-in, array-out) and
mirror the MATLAB functions in  matlab/  with identical sign conventions and
parameter names.

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
# Main implicit-spline function
# ──────────────────────────────────────────────────────────────────────────────

def imp_spline_2d(x, y, P, delta: float = 0.1, n: int = 2) -> np.ndarray:
    """2D piecewise algebraic implicit spline for polygon *P*.

    Evaluates the smooth implicit function at query point(s) ``(x, y)``
    using the per-edge product construction:

    .. math::

        f(x,y) = \\prod_{i=0}^{m-1}  H\\!\\left(L_i(x,y),\\, \\delta,\\, n\\right)

    where :math:`L_i` is the normalised signed distance from ``(x,y)`` to
    edge *i* (positive on the interior side).  The result is:

    * ``f ≈ 1``  deep inside the polygon (all :math:`L_i \\geq \\delta`)
    * ``f = 0``  on the polygon boundary (at least one :math:`L_i = 0`)
    * ``f ≈ 0``  outside the polygon (at least one :math:`L_i \\leq 0`)

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
    The construction is exact for convex polygons.  For non-convex polygons,
    regions near reflex vertices may receive lower values; increase *delta*
    or use a convex decomposition.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    >>> imp_spline_2d(0.5, 0.5, P, delta=0.1, n=2)
    array(1.)
    >>> imp_spline_2d(0.0, 0.5, P, delta=0.1, n=2)
    array(0.)

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

    # Ensure counter-clockwise orientation so that L_i > 0 inside polygon
    if polygon_signed_area(P) < 0:
        P = P[::-1]

    # Product of per-edge smooth step functions
    # BUG NOTE: the original MATLAB code used an undeclared 'zz' at this
    # point instead of the output variable; that is corrected here by using
    # the consistently named accumulator 'f'.
    f = np.ones_like(x)
    for i in range(m):
        j = (i + 1) % m
        L_i = lxy(x, y, P[i, 0], P[i, 1], P[j, 0], P[j, 1])
        f = f * H(L_i, delta, n)

    return f
