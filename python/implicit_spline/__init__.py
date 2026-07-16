"""
implicit_spline
===============
2D Piecewise Algebraic Splines for Implicit Modeling.

Implements the construction from:
    Li, Q. & Tian, J. (2009).  2D Piecewise Algebraic Splines for Implicit
    Modeling.  ACM Transactions on Graphics, 28(3).
    DOI: 10.1145/1516522.1516524

Main API
--------
imp_spline_2d(x, y, P, delta, n)
    Evaluate the implicit spline at point(s) (x, y) for polygon P.

draw_imp_spline(P, ...)
    Evaluate on a grid and produce Matplotlib figures.

H(t, delta, n)
    Smooth Heaviside-like step function (building block).

lxy(x, y, x1, y1, x2, y2)
    Normalised signed distance from (x, y) to directed line.

Quick start
-----------
>>> import numpy as np
>>> from implicit_spline import imp_spline_2d, draw_imp_spline
>>> P = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
>>> draw_imp_spline(P, delta=0.1, n=2)
"""

from .core import (
    H,
    lxy,
    lxy00,
    point_imp,
    polygon_signed_area,
    imp_spline_2d,
    smooth_union,
    convex_decomp_field,
    partition_basis_normalized,
)
from .visualization import (
    make_grid,
    draw_imp_spline,
    draw_surface,
    compare_delta,
)

__all__ = [
    "H",
    "lxy",
    "lxy00",
    "point_imp",
    "polygon_signed_area",
    "imp_spline_2d",
    "smooth_union",
    "convex_decomp_field",
    "partition_basis_normalized",
    "make_grid",
    "draw_imp_spline",
    "draw_surface",
    "compare_delta",
]

__version__ = "0.1.0"
