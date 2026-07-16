"""
implicit_spline
===============
2D Piecewise Algebraic Splines for Implicit Modeling.
"""

from .core import (
    H,
    lxy,
    lxy00,
    point_imp,
    polygon_signed_area,
    is_convex,
    polygon_validate,
    triangulate_polygon,
    cancel_internal_edges,
    convex_product_spline_2d,
    imp_spline_2d,
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

__version__ = "0.1.0"
