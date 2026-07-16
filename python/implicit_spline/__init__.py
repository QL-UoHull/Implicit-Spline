"""
implicit_spline
===============
2D Piecewise Algebraic Splines for Implicit Modeling.

Python port of the MATLAB reference implementation (Li & Tian, 2009).
"""

from .core import (
    # MATLAB primitives (H.m, Lxy.m, Lxy00.m, Point_imp.m)
    H,
    lxy,
    lxy00,
    point_imp,
    # MATLAB primitives (LineSeg_imp.m, L_corner_inter.m, U_Angle_inter.m,
    #                     Square_Angle_inter.m)
    line_seg_imp,
    l_corner_inter,
    u_angle_inter,
    square_angle_inter,
    # Polygon utilities
    polygon_signed_area,
    is_convex,
    polygon_validate,
    triangulate_polygon,
    cancel_internal_edges,
    # Evaluators
    convex_product_spline_2d,   # MATLAB ImpSpline2D (product formula, convex only)
    imp_spline_2d,              # H-kernel boundary integral (arbitrary polygon)
    # Composition helpers
    smooth_union,
    convex_decomp_field,
    partition_basis_fields,
    partition_basis_normalized,
    validate_partition,
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
    "line_seg_imp",
    "l_corner_inter",
    "u_angle_inter",
    "square_angle_inter",
    "polygon_signed_area",
    "is_convex",
    "polygon_validate",
    "triangulate_polygon",
    "cancel_internal_edges",
    "convex_product_spline_2d",
    "imp_spline_2d",
    "smooth_union",
    "convex_decomp_field",
    "partition_basis_fields",
    "partition_basis_normalized",
    "validate_partition",
    "make_grid",
    "draw_imp_spline",
    "draw_surface",
    "compare_delta",
]

__version__ = "0.2.0"
