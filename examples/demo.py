"""
demo.py
-------
Standalone Python demonstration of the 2D Piecewise Algebraic Implicit Spline.

This demo is organized around the three main modeling scenarios highlighted in
Li & Tian (2009):

  1. Convex polygons
  2. Polygons with holes
  3. Collections of polygons forming a 2D partition

It also includes paper-style multi-panel figures inspired by the examples shown
in the paper.

Run from the repository root::

    python examples/demo.py

or from within the examples/ directory::

    python demo.py

Reference
---------
Li, Q. & Tian, J. (2009). 2D Piecewise Algebraic Splines for Implicit
Modeling. ACM Transactions on Graphics, 28(3).
DOI: 10.1145/1516522.1516524
"""

import sys
import os

# Allow running from examples/ or from the repo root
_here = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.join(_here, '..', 'python')
if os.path.isdir(_python_dir):
    sys.path.insert(0, os.path.abspath(_python_dir))

import numpy as np
import matplotlib.pyplot as plt

from implicit_spline import imp_spline_2d, draw_imp_spline
from implicit_spline.visualization import (
    compare_delta,
    compare_n,
    panel_delta_shapes,
    partition_basis_surfaces,
    draw_polygon_outline,
    make_grid,
)

# ── Shared parameters ─────────────────────────────────────────────────────────
DELTA = 0.12
N_ORDER = 2
N_GRID = 260


def plot_polygon_with_holes(outer, holes, delta=0.15, n=2, N=320, title=None):
    """Visualize an implicit shape defined by an outer polygon minus hole polygons."""
    loops = [np.asarray(outer, dtype=float)] + [np.asarray(h, dtype=float) for h in holes]
    all_pts = np.vstack(loops)
    X, Y = make_grid(all_pts, N=N, pad_fraction=0.20)

    Z = imp_spline_2d(X, Y, outer, delta=delta, n=n)
    for hole in holes:
        Z = Z * (1.0 - imp_spline_2d(X, Y, hole, delta=delta, n=n))
    Z = np.clip(Z, 0.0, 1.0)

    fig = plt.figure(figsize=(11, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    cf = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax1)
    ax1.contour(X, Y, Z, levels=[0.5], colors='white', linewidths=2)
    draw_polygon_outline(outer, ax=ax1, linestyle=':', color='0.55', linewidth=0.9,
                         marker='o', markersize=3)
    for hole in holes:
        draw_polygon_outline(hole, ax=ax1, linestyle=':', color='0.40', linewidth=0.9,
                             marker='o', markersize=3)
    ax1.set_title('Contour with holes')

    ax2.plot_wireframe(X, Y, Z, rstride=5, cstride=5, color='0.35', linewidth=0.45)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    ax2.set_zlim(0.0, 1.05)
    ax2.view_init(elev=28, azim=-58)
    ax2.set_title('Wireframe surface')

    if title is not None:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


print("=== 2D Implicit Spline Demo ===\n")

# ============================================================================
# 1. CONVEX POLYGONS
# ============================================================================
print("Category 1: convex polygons")

# Example 1: Unit square
print("  Example 1: unit square")
P_sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
fig1, ax1 = plt.subplots(figsize=(6, 5))
fig1.canvas.manager.set_window_title("Demo — Convex polygon: square")
draw_imp_spline(P_sq, delta=DELTA, n=N_ORDER, N=N_GRID, ax=ax1,
                title=rf"Convex polygon: square ($\delta={DELTA}$, $n={N_ORDER}$)")
plt.tight_layout()

# Example 2: Equilateral triangle
print("  Example 2: equilateral triangle")
P_tri = np.array([[0, 0], [2, 0], [1, np.sqrt(3)]], dtype=float)
fig2, ax2 = plt.subplots(figsize=(6, 5))
fig2.canvas.manager.set_window_title("Demo — Convex polygon: triangle")
draw_imp_spline(P_tri, delta=DELTA, n=N_ORDER, N=N_GRID, ax=ax2,
                title=rf"Convex polygon: triangle ($\delta={DELTA}$, $n={N_ORDER}$)")
plt.tight_layout()

# Example 3: Regular pentagon
print("  Example 3: regular pentagon")
theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 6)[:-1]
P_pent = np.column_stack([np.cos(theta), np.sin(theta)])
fig3, ax3 = plt.subplots(figsize=(6, 5))
fig3.canvas.manager.set_window_title("Demo — Convex polygon: pentagon")
draw_imp_spline(P_pent, delta=DELTA, n=N_ORDER, N=N_GRID, ax=ax3,
                title=rf"Convex polygon: pentagon ($\delta={DELTA}$, $n={N_ORDER}$)")
plt.tight_layout()

# Example 4: Delta sweep on an irregular convex polygon
print("  Example 4: convex polygon delta sweep")
P_convex = np.array([[0, 0], [2, 0], [2.5, 0.8], [1.5, 1.8], [-0.2, 1.2]], dtype=float)
fig4 = compare_delta(P_convex, deltas=(0.05, 0.15, 0.30), n=N_ORDER, N=N_GRID)
fig4.canvas.manager.set_window_title("Demo — Convex polygon delta sweep")

# Example 5: Smoothness order sweep
print("  Example 5: convex polygon smoothness sweep")
fig5 = compare_n(P_sq, delta=0.15, n_values=(1, 2, 3), N=N_GRID)
fig5.canvas.manager.set_window_title("Demo — Convex polygon smoothness sweep")

# Example 6: Paper-style multi-panel figure for a convex control polygon
print("  Example 6: paper-style delta panel for a convex polygon")
P_ctrl = np.array([
    [-0.95, -0.78],
    [-0.90,  0.80],
    [-0.08,  0.40],
    [ 0.72,  0.84],
    [ 1.02, -0.80],
    [ 0.25, -0.45],
    [ 0.00, -0.60],
    [-0.38, -0.58],
], dtype=float)
fig6 = panel_delta_shapes(
    P_ctrl,
    deltas=(0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
    n=2,
    N=320,
    layout=(2, 3),
    title="Convex polygon: contour evolution under increasing $\\delta$",
)
fig6.canvas.manager.set_window_title("Demo — Convex polygon paper-style delta panel")

# ============================================================================
# 2. POLYGONS WITH HOLES
# ============================================================================
print("Category 2: polygons with holes")

# Example 7: Triangle with inner triangular hole
print("  Example 7: triangle with a hole")
outer_tri = np.array([[-1.10, -0.78], [-0.10, 0.72], [1.10, -0.78]], dtype=float)
inner_tri = np.array([[-0.25, -0.36], [0.15, -0.45], [-0.10, 0.05]], dtype=float)
fig7 = plot_polygon_with_holes(
    outer_tri,
    [inner_tri],
    delta=0.12,
    n=2,
    N=320,
    title="Polygon with hole: outer triangle and inner triangular void",
)
fig7.canvas.manager.set_window_title("Demo — Polygon with hole: triangle")

# Example 8: Rounded rectangular outer boundary with two holes
print("  Example 8: polygon with multiple holes")
outer_rect = np.array([
    [-1.15, -0.82], [-1.00, 0.88], [1.15, 0.92], [1.20, -0.82]
], dtype=float)
left_hole = np.array([[-0.78, 0.36], [-0.48, -0.28], [0.10, 0.48]], dtype=float)
right_hole = np.array([[0.42, -0.28], [0.98, 0.18], [1.02, -0.56]], dtype=float)
fig8 = plot_polygon_with_holes(
    outer_rect,
    [left_hole, right_hole],
    delta=0.18,
    n=2,
    N=340,
    title="Polygon with holes: outer boundary with two internal voids",
)
fig8.canvas.manager.set_window_title("Demo — Polygon with holes: multiple")

# Example 9: Outer polygon with a star-like central hole and delta sweep
print("  Example 9: hole-shape evolution under varying delta")
outer_loop = np.array([
    [-1.10, -0.92], [-1.12, 0.95], [1.18, 0.95], [1.20, -0.95]
], dtype=float)
inner_loop = np.array([
    [-0.55, 0.05], [-0.15, 0.35], [0.15, 0.02], [0.52, 0.32],
    [0.80, -0.05], [0.38, -0.28], [0.18, -0.58], [-0.12, -0.48],
    [-0.38, -0.62], [-0.68, -0.22],
], dtype=float)
fig9, axes9 = plt.subplots(2, 2, figsize=(10.5, 8.2), squeeze=False)
all_pts = np.vstack([outer_loop, inner_loop])
X, Y = make_grid(all_pts, N=320, pad_fraction=0.18)
for ax, d in zip(axes9.ravel(), (0.08, 0.15, 0.28, 0.40)):
    Z = imp_spline_2d(X, Y, outer_loop, delta=d, n=2)
    Z = Z * (1.0 - imp_spline_2d(X, Y, inner_loop, delta=d, n=2))
    Z = np.clip(Z, 0.0, 1.0)
    ax.contour(X, Y, Z, levels=[0.5], colors='k', linewidths=0.85)
    draw_polygon_outline(outer_loop, ax=ax, linestyle=':', color='0.65', linewidth=0.8,
                         marker='*', markersize=5)
    draw_polygon_outline(inner_loop, ax=ax, linestyle=':', color='0.45', linewidth=0.8,
                         marker='*', markersize=5)
    ax.set_title(rf"$\delta$={d}")
fig9.suptitle("Polygon with hole: implicit contour evolution under increasing $\\delta$", fontsize=13)
plt.tight_layout()
fig9.canvas.manager.set_window_title("Demo — Polygon with hole delta panel")

# ============================================================================
# 3. COLLECTION OF POLYGONS FORMING A 2D PARTITION
# ============================================================================
print("Category 3: collections of polygons forming a 2D partition")

# Example 10: Partition net and summed basis surfaces
print("  Example 10: partition net and summed basis surfaces")
partition_polygons = [
    np.array([[-0.95, -0.42], [-0.55,  0.10], [-0.55, -1.00]]),
    np.array([[-0.95,  0.45], [-0.55,  0.10], [-0.55,  0.85], [-0.35, 0.98], [-0.02, 0.62], [-0.32, 0.12]]),
    np.array([[-0.55,  0.10], [-0.02, 0.62], [0.48, 0.78], [0.72, 0.55], [0.48, 0.12], [0.12, -0.02], [-0.32, 0.12]]),
    np.array([[-0.32, 0.12], [0.12, -0.02], [0.35, -0.42], [-0.15, -0.62], [-0.55, -0.55], [-0.55, 0.10]]),
    np.array([[0.12, -0.02], [0.48, 0.12], [0.70, -0.30], [0.35, -0.42]]),
    np.array([[0.48, 0.12], [0.72, 0.55], [1.05, 0.42], [0.85, -0.18], [0.70, -0.30]]),
    np.array([[0.35, -0.42], [0.70, -0.30], [0.85, -0.18], [0.62, -0.92], [0.18, -0.98], [-0.15, -0.62]]),
]
fig10 = partition_basis_surfaces(partition_polygons, deltas=(0.05, 0.10, 0.20), n=2, N=160)
fig10.canvas.manager.set_window_title("Demo — Partition net and basis surfaces")

# Example 11: Individual partition-cell basis gallery
print("  Example 11: basis gallery for selected partition cells")
fig11, axes11 = plt.subplots(2, 2, figsize=(10.5, 8.0), squeeze=False)
for idx, (ax, poly) in enumerate(zip(axes11.ravel(), partition_polygons[:4]), start=1):
    draw_imp_spline(poly, delta=0.12, n=2, N=260, ax=ax,
                    title=f"Partition cell {idx}")
fig11.suptitle("Basis functions associated with individual partition polygons", fontsize=13)
plt.tight_layout()
fig11.canvas.manager.set_window_title("Demo — Partition cell basis gallery")

# Example 12: Freeform curves from a small polygon collection
print("  Example 12: freeform shapes from multiple polygon components")
collection_shapes = [
    np.array([[-1.00, -0.70], [-0.70, 0.20], [-0.30, 0.72], [0.05, 0.30], [-0.15, -0.62]]),
    np.array([[0.10, -0.42], [0.62, 0.20], [1.00, -0.58], [0.82, -0.88], [0.30, -0.82]]),
    np.array([[-0.12, 0.15], [0.20, 0.72], [0.58, 0.18], [0.32, -0.18]]),
]
all_pts = np.vstack(collection_shapes)
Xc, Yc = make_grid(all_pts, N=320, pad_fraction=0.20)
Zc = np.zeros_like(Xc)
for poly in collection_shapes:
    Zc += imp_spline_2d(Xc, Yc, poly, delta=0.16, n=2)

fig12 = plt.figure(figsize=(11, 4.2))
ax12a = fig12.add_subplot(1, 2, 1)
ax12b = fig12.add_subplot(1, 2, 2, projection='3d')
cf12 = ax12a.contourf(Xc, Yc, Zc, levels=20, cmap='viridis')
plt.colorbar(cf12, ax=ax12a)
ax12a.contour(Xc, Yc, Zc, levels=[0.5, 1.0], colors=['white', 'black'], linewidths=[2.0, 1.0])
for poly in collection_shapes:
    draw_polygon_outline(poly, ax=ax12a, linestyle=':', color='0.55', linewidth=0.8,
                         marker='o', markersize=3)
ax12a.set_title('Combined contour field')
ax12b.plot_wireframe(Xc, Yc, Zc, rstride=5, cstride=5, color='0.35', linewidth=0.45)
ax12b.set_xlabel('x')
ax12b.set_ylabel('y')
ax12b.set_zlabel('sum')
ax12b.view_init(elev=28, azim=-58)
ax12b.set_title('Summed wireframe surface')
fig12.suptitle('Collection of polygons: composed implicit field', fontsize=13)
plt.tight_layout()
fig12.canvas.manager.set_window_title("Demo — Collection of polygons")

print("\nDemo complete.")
plt.show()
