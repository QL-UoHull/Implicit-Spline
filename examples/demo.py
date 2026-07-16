"""
demo.py
-------
Standalone Python demonstration of the 2D Piecewise Algebraic Implicit Spline.

This demo extends the paper's core scenarios and is organized into three
feature-focused sections:

  1. Convex polygons
  2. Polygons with holes (composed implicit fields)
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

# Example 7: Freeform silhouette via composition of convex components
print("  Example 7: freeform silhouette via convex decomposition (composed field)")
convex_components = [
    np.array([[-1.00, -0.52], [-0.86, 0.48], [-0.30, 0.78], [0.05, 0.32], [-0.20, -0.58]], dtype=float),
    np.array([[0.00, -0.45], [0.35, 0.18], [0.95, -0.50], [0.80, -0.85], [0.22, -0.80]], dtype=float),
    np.array([[-0.10, 0.12], [0.15, 0.70], [0.58, 0.20], [0.30, -0.15]], dtype=float),
]
all_cmp_pts = np.vstack(convex_components)
X7, Y7 = make_grid(all_cmp_pts, N=320, pad_fraction=0.20)
Z7 = np.zeros_like(X7)
for poly in convex_components:
    Z7 += imp_spline_2d(X7, Y7, poly, delta=0.16, n=2)

fig7 = plt.figure(figsize=(11, 4.2))
ax7a = fig7.add_subplot(1, 2, 1)
ax7b = fig7.add_subplot(1, 2, 2, projection='3d')
cf7 = ax7a.contourf(X7, Y7, Z7, levels=20, cmap='viridis')
plt.colorbar(cf7, ax=ax7a)
ax7a.contour(X7, Y7, Z7, levels=[0.5, 1.0], colors=['white', 'black'], linewidths=[2.0, 1.0])
for poly in convex_components:
    draw_polygon_outline(poly, ax=ax7a, linestyle=':', color='0.55', linewidth=0.8,
                         marker='o', markersize=3)
ax7a.set_title('Composed contour from convex parts')
ax7b.plot_wireframe(X7, Y7, Z7, rstride=5, cstride=5, color='0.35', linewidth=0.45)
ax7b.set_xlabel('x')
ax7b.set_ylabel('y')
ax7b.set_zlabel('sum')
ax7b.view_init(elev=28, azim=-58)
ax7b.set_title('Composed wireframe surface')
fig7.suptitle('Freeform-like shape as composed convex implicit fields', fontsize=13)
plt.tight_layout()
fig7.canvas.manager.set_window_title("Demo — Composed freeform from convex components")

# ============================================================================
# 2. POLYGONS WITH HOLES
# ============================================================================
print("Category 2: polygons with holes (composed fields)")

# Example 8: Triangle with inner triangular hole
print("  Example 8: triangle with a hole (outer × complements of holes)")
outer_tri = np.array([[-1.10, -0.78], [-0.10, 0.72], [1.10, -0.78]], dtype=float)
inner_tri = np.array([[-0.25, -0.36], [0.15, -0.45], [-0.10, 0.05]], dtype=float)
fig8 = plot_polygon_with_holes(
    outer_tri,
    [inner_tri],
    delta=0.12,
    n=2,
    N=320,
    title="Composed hole field: outer triangle × (1 - inner triangle field)",
)
fig8.canvas.manager.set_window_title("Demo — Polygon with hole: triangle")

# Example 9: Convex outer boundary with two convex holes
print("  Example 9: polygon with multiple convex holes")
outer_rect = np.array([
    [-1.15, -0.82], [-1.00, 0.88], [1.15, 0.92], [1.20, -0.82]
], dtype=float)
left_hole = np.array([[-0.78, 0.36], [-0.48, -0.28], [0.10, 0.48]], dtype=float)
right_hole = np.array([[0.42, -0.28], [0.98, 0.18], [1.02, -0.56]], dtype=float)
fig9 = plot_polygon_with_holes(
    outer_rect,
    [left_hole, right_hole],
    delta=0.18,
    n=2,
    N=340,
    title="Composed hole field: convex outer loop with two convex inner loops",
)
fig9.canvas.manager.set_window_title("Demo — Polygon with holes: multiple")

# Example 10: Convex loop-with-hole delta sweep
print("  Example 10: hole-shape evolution under varying delta (convex loops)")
outer_loop = np.array([
    [-1.08, -0.90], [-1.15, 0.70], [-0.25, 1.05], [0.85, 0.88], [1.18, -0.10], [0.40, -1.00]
], dtype=float)
inner_loop = np.array([
    [-0.36, -0.10], [-0.02, 0.32], [0.38, 0.06], [0.08, -0.38]
], dtype=float)
fig10, axes10 = plt.subplots(2, 2, figsize=(10.5, 8.2), squeeze=False)
all_pts = np.vstack([outer_loop, inner_loop])
X, Y = make_grid(all_pts, N=320, pad_fraction=0.18)
for ax, d in zip(axes10.ravel(), (0.08, 0.15, 0.28, 0.40)):
    Z = imp_spline_2d(X, Y, outer_loop, delta=d, n=2)
    Z = Z * (1.0 - imp_spline_2d(X, Y, inner_loop, delta=d, n=2))
    Z = np.clip(Z, 0.0, 1.0)
    ax.contour(X, Y, Z, levels=[0.5], colors='k', linewidths=0.85)
    draw_polygon_outline(outer_loop, ax=ax, linestyle=':', color='0.65', linewidth=0.8,
                         marker='*', markersize=5)
    draw_polygon_outline(inner_loop, ax=ax, linestyle=':', color='0.45', linewidth=0.8,
                         marker='*', markersize=5)
    ax.set_title(rf"$\delta$={d}")
fig10.suptitle("Composed loop-with-hole contours under increasing $\\delta$", fontsize=13)
plt.tight_layout()
fig10.canvas.manager.set_window_title("Demo — Polygon with hole delta panel")

# ============================================================================
# 3. COLLECTION OF POLYGONS FORMING A 2D PARTITION
# ============================================================================
print("Category 3: collections of polygons forming a 2D partition")

# Example 11: Convex-cell partition and aggregate cell-field surfaces
print("  Example 11: convex-cell partition net and aggregate cell-field surfaces")
x_nodes = np.linspace(-1.2, 1.2, 4)
y_nodes = np.linspace(-1.2, 1.2, 4)
partition_polygons = []
for i in range(len(x_nodes) - 1):
    for j in range(len(y_nodes) - 1):
        x0, x1 = x_nodes[i], x_nodes[i + 1]
        y0, y1 = y_nodes[j], y_nodes[j + 1]
        partition_polygons.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float))

fig11 = partition_basis_surfaces(partition_polygons, deltas=(0.05, 0.10, 0.20), n=2, N=160)
fig11.suptitle("Convex-cell partition: net and unnormalized sums of cell fields", fontsize=13)
plt.tight_layout()
fig11.canvas.manager.set_window_title("Demo — Convex-cell partition and aggregate fields")

# Example 12: Individual partition-cell basis gallery
print("  Example 12: basis gallery for selected partition cells")
fig12, axes12 = plt.subplots(2, 2, figsize=(10.5, 8.0), squeeze=False)
selected_cells = [partition_polygons[0], partition_polygons[1], partition_polygons[4], partition_polygons[8]]
for idx, (ax, poly) in enumerate(zip(axes12.ravel(), selected_cells), start=1):
    draw_imp_spline(poly, delta=0.12, n=2, N=260, ax=ax,
                    title=f"Partition cell {idx}")
fig12.suptitle("Cell-wise implicit fields on selected convex partition cells", fontsize=13)
plt.tight_layout()
fig12.canvas.manager.set_window_title("Demo — Partition cell basis gallery")

# Example 13: Aggregate contour and wireframe over the partition cells
print("  Example 13: aggregate field over all partition cells (unnormalized sum)")
all_pts = np.vstack(partition_polygons)
Xc, Yc = make_grid(all_pts, N=320, pad_fraction=0.08)
Zc = np.zeros_like(Xc)
for poly in partition_polygons:
    Zc += imp_spline_2d(Xc, Yc, poly, delta=0.16, n=2)

fig13 = plt.figure(figsize=(11, 4.2))
ax13a = fig13.add_subplot(1, 2, 1)
ax13b = fig13.add_subplot(1, 2, 2, projection='3d')
cf13 = ax13a.contourf(Xc, Yc, Zc, levels=20, cmap='viridis')
plt.colorbar(cf13, ax=ax13a)
ax13a.contour(Xc, Yc, Zc, levels=[0.5, 1.0], colors=['white', 'black'], linewidths=[2.0, 1.0])
for poly in partition_polygons:
    draw_polygon_outline(poly, ax=ax13a, linestyle=':', color='0.55', linewidth=0.8,
                         marker='o', markersize=3)
ax13a.set_title('Aggregate contour field (sum over cells)')
ax13b.plot_wireframe(Xc, Yc, Zc, rstride=5, cstride=5, color='0.35', linewidth=0.45)
ax13b.set_xlabel('x')
ax13b.set_ylabel('y')
ax13b.set_zlabel('sum')
ax13b.view_init(elev=28, azim=-58)
ax13b.set_title('Aggregate wireframe surface')
fig13.suptitle('Convex-cell partition: aggregate implicit field (not normalized)', fontsize=13)
plt.tight_layout()
fig13.canvas.manager.set_window_title("Demo — Partition aggregate field")

print("\nDemo complete.")
plt.show()
