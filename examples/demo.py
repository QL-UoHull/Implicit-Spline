"""Standalone demonstration of the corrected additive implicit-spline construction.

Sections
--------
1. Convex polygon example
2. Polygon-with-hole composition
3. Section 7: two paper-style non-convex polygons (direct and decomposition)
4. Section 9: conforming irregular polygon partition and additive basis identity
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg" if os.environ.get("DISPLAY", "") == "" else matplotlib.get_backend())
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from implicit_spline.core import (
    convex_decomp_field,
    imp_spline_2d,
    partition_basis_fields,
    polygon_validate,
    triangulate_polygon,
    validate_partition,
)
from implicit_spline.paper_examples import (
    PARTITION_CELLS,
    PARTITION_OUTER,
    SECTION7_POLYGONS,
)
from implicit_spline.visualization import _safe_contour, make_grid

DELTA = 0.22
N_ORDER = 2
GRID_N = 220


def _close_polygon(P):
    return np.vstack([P, P[0]])


def _paper_axes(ax, P, *, xpad=0.35, ypad=0.35):
    P = np.asarray(P, dtype=float)
    ax.set_aspect("equal")
    ax.set_xlim(P[:, 0].min() - xpad, P[:, 0].max() + xpad)
    ax.set_ylim(P[:, 1].min() - ypad, P[:, 1].max() + ypad)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_control_polygon(ax, P):
    Pc = _close_polygon(P)
    ax.plot(Pc[:, 0], Pc[:, 1], "--", color="0.55", linewidth=0.9)
    ax.plot(P[:, 0], P[:, 1], "o", color="0.35", markersize=3.2)


def _draw_iso(ax, P, *, delta=DELTA, n=N_ORDER, title=None):
    X, Y = make_grid(P, N=GRID_N, pad_fraction=0.12)
    Z = imp_spline_2d(X, Y, P, delta=delta, n=n)
    _safe_contour(ax, X, Y, Z, levels=[0.5], colors="k", linewidths=0.95)
    _draw_control_polygon(ax, P)
    _paper_axes(ax, P)
    if title:
        ax.set_title(title, fontsize=10)
    return X, Y, Z


def figure_convex_and_hole_examples():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    square = np.array([[-1.2, -1.0], [1.2, -1.0], [1.5, 0.8], [-0.8, 1.4]], dtype=float)
    _draw_iso(axes[0], square, delta=0.18, title="Convex polygon")

    outer = np.array([[-2.1, -1.5], [2.0, -1.6], [2.3, 1.0], [0.0, 2.2], [-2.3, 0.9]], dtype=float)
    hole = np.array([[-0.55, -0.35], [0.55, -0.35], [0.55, 0.55], [-0.55, 0.55]], dtype=float)
    X, Y = make_grid(outer, N=GRID_N, pad_fraction=0.12)
    Z = imp_spline_2d(X, Y, outer, delta=0.16, n=N_ORDER) * (1.0 - imp_spline_2d(X, Y, hole, delta=0.16, n=N_ORDER))
    _safe_contour(axes[1], X, Y, Z, levels=[0.5], colors="k", linewidths=0.95)
    _draw_control_polygon(axes[1], outer)
    _draw_control_polygon(axes[1], hole)
    _paper_axes(axes[1], outer)
    axes[1].set_title("Polygon with hole", fontsize=10)

    fig.suptitle("Implicit spline examples", fontsize=12)
    fig.tight_layout()
    return fig


def figure_section7_main():
    names = ["heart_like", "narrow_neck"]
    titles = ["Section 7(a): shallow concave upper chain", "Section 7(b): narrow neck / broad base"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    diagnostics = []

    for ax, name, title in zip(axes, names, titles):
        P = polygon_validate(SECTION7_POLYGONS[name], name=name)
        X, Y, Z = _draw_iso(ax, P, title=title)
        tris = triangulate_polygon(P)
        Zd = convex_decomp_field(X, Y, tris, delta=DELTA, n=N_ORDER)
        diagnostics.append((name, float(np.max(np.abs(Z - Zd)))))

    fig.tight_layout()
    return fig, diagnostics


def figure_section9_partition():
    validate_partition(PARTITION_CELLS, PARTITION_OUTER)
    fig = plt.figure(figsize=(11, 4.8))
    ax_net = fig.add_subplot(1, 2, 1)
    ax_sum = fig.add_subplot(1, 2, 2, projection="3d")

    for cell in PARTITION_CELLS:
        Pc = _close_polygon(cell)
        ax_net.plot(Pc[:, 0], Pc[:, 1], "-", color="0.15", linewidth=0.9)
    _paper_axes(ax_net, PARTITION_OUTER, xpad=0.2, ypad=0.2)
    ax_net.set_title("Section 9: partition net", fontsize=10)

    X, Y = make_grid(PARTITION_OUTER, N=140, pad_fraction=0.06)
    basis, total = partition_basis_fields(PARTITION_CELLS, X, Y, delta=0.18, n=N_ORDER)
    outer = imp_spline_2d(X, Y, PARTITION_OUTER, delta=0.18, n=N_ORDER)
    ax_sum.plot_wireframe(X, Y, total, rstride=4, cstride=4, color="0.25", linewidth=0.45)
    ax_sum.set_xlabel("x")
    ax_sum.set_ylabel("y")
    ax_sum.set_zlabel(r"$\sum_k B_k$")
    ax_sum.set_title(r"Additive identity: $\,\sum_k B_k = B_{\Omega}$", fontsize=10)
    ax_sum.view_init(elev=28, azim=-58)
    fig.tight_layout()

    return fig, float(np.max(np.abs(total - outer))), len(basis)


def main():
    fig1 = figure_convex_and_hole_examples()
    fig2, section7_diag = figure_section7_main()
    fig3, partition_err, n_cells = figure_section9_partition()

    print("Corrected additive implicit-spline demo")
    print(f"  delta={DELTA}, n={N_ORDER}")
    for name, err in section7_diag:
        print(f"  Section 7 direct vs decomposition ({name}): max|Δ| = {err:.3e}")
    print(f"  Section 9 partition cells: {n_cells}")
    print(f"  Section 9 additive sum vs outer boundary basis: max|Δ| = {partition_err:.3e}")

    out_dir = os.path.join(REPO_ROOT, "examples")
    fig1.savefig(os.path.join(out_dir, "demo_basic_examples.png"), dpi=180, bbox_inches="tight")
    fig2.savefig(os.path.join(out_dir, "demo_section7_paper_style.png"), dpi=180, bbox_inches="tight")
    fig3.savefig(os.path.join(out_dir, "demo_section9_partition.png"), dpi=180, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
