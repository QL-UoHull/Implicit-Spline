"""
demo.py
-------
Standalone Python demonstration of the 2D Piecewise Algebraic Implicit Spline.

This demo is organized into three feature-focused sections:

  1. Convex polygons
  2. Polygons with holes (composed implicit fields)
  3. Concave polygon via convex decomposition  ← Section 7 (corrected)
  4. Collections of convex polygons forming a 2D partition  ← Section 9 (corrected)

Design notes
------------
**Concave polygons (Section 7)**
  The core function ``imp_spline_2d`` is a *product* of smooth half-plane
  fields—one per polygon edge—so it is exact for *convex* polygons only.
  Feeding a concave polygon directly gives incorrect results near reflex
  vertices (the interior product can be zero there).

  The correct approach is a convex decomposition:

  1. Provide the concave polygon with vertices in **CCW boundary order**
     (verified via ``polygon_signed_area > 0``).
  2. Decompose it into non-overlapping convex pieces.
  3. Combine piece fields with the **bounded smooth union**
     ``B = 1 − ∏_k (1 − B_k)`` (``convex_decomp_field``), which stays in
     [0, 1] unlike a raw sum.

**Partition basis (Section 9)**
  A raw sum Σ B_k over partition cells is *not* a partition of unity—it
  can exceed 1 near cell interiors and falls below 1 near shared edges.
  The correct partition basis normalizes each cell field:
  ``B̂_k = B_k / max(Σ_j B_j, ε)``, which satisfies Σ_k B̂_k = 1 exactly
  wherever the denominator exceeds ε.

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

from implicit_spline import (
    imp_spline_2d,
    draw_imp_spline,
    polygon_signed_area,
    convex_decomp_field,
    partition_basis_normalized,
)
from implicit_spline.visualization import (
    compare_delta,
    compare_n,
    panel_delta_shapes,
    partition_basis_surfaces,
    draw_polygon_outline,
    make_grid,
    _safe_contour,
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
    _safe_contour(ax1, X, Y, Z, levels=[0.5], colors='white', linewidths=2)
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
    _safe_contour(ax, X, Y, Z, levels=[0.5], colors='k', linewidths=0.85)
    draw_polygon_outline(outer_loop, ax=ax, linestyle=':', color='0.65', linewidth=0.8,
                         marker='*', markersize=5)
    draw_polygon_outline(inner_loop, ax=ax, linestyle=':', color='0.45', linewidth=0.8,
                         marker='*', markersize=5)
    ax.set_title(rf"$\delta$={d}")
fig10.suptitle("Composed loop-with-hole contours under increasing $\\delta$", fontsize=13)
plt.tight_layout()
fig10.canvas.manager.set_window_title("Demo — Polygon with hole delta panel")

# ============================================================================
# 3. CONCAVE POLYGON VIA CONVEX DECOMPOSITION  (Section 7)
# ============================================================================
# ── Why CCW order alone is insufficient ──────────────────────────────────────
#
# ``imp_spline_2d`` computes a *product* of smooth half-plane fields, one per
# edge.  At a reflex vertex the interior is on the "wrong" side of one of the
# adjacent half-planes, so the product is 0 there even though the point is
# inside the polygon.  Reordering vertices to CCW corrects the *winding* but
# cannot repair this topological deficiency.
#
# The correct representation uses a *convex decomposition*:
#   1. Express the concave polygon as a union of convex pieces.
#   2. Evaluate ``imp_spline_2d`` on each convex piece.
#   3. Combine with the bounded smooth union  B = 1 − ∏_k (1 − B_k),
#      implemented by ``convex_decomp_field``.
#      This formula stays in [0, 1] (unlike a raw sum), maps to 0 outside
#      all pieces, and to ≈ 1 deep inside any piece.
# ─────────────────────────────────────────────────────────────────────────────
print("Category 3: concave polygon via convex decomposition")

# Canonical concave example: L-shaped polygon.
# Vertices are listed explicitly in counter-clockwise (CCW) boundary order.
#
#   (0,2)──(1,2)
#   │       │
#   │  P2   │   ← upper 1×1 square
#   │       │
#   (0,1)──(1,1)──(2,1)
#   │               │
#   │     P1        │   ← lower 2×1 rectangle
#   │               │
#   (0,0)──────────(2,0)
#
P_concave = np.array([
    [0.0, 0.0],   # P0: bottom-left
    [2.0, 0.0],   # P1: bottom-right
    [2.0, 1.0],   # P2: right-mid (reflex-adjacent)
    [1.0, 1.0],   # P3: inner corner  ← reflex vertex of the concave boundary
    [1.0, 2.0],   # P4: top-right
    [0.0, 2.0],   # P5: top-left
], dtype=float)

# ── Orientation check ────────────────────────────────────────────────────────
_area_concave = polygon_signed_area(P_concave)
assert _area_concave > 0, (
    f"P_concave must be in CCW order (positive signed area); got {_area_concave:.6f}. "
    "To fix, reverse the vertex order: P_concave = P_concave[::-1]"
)
print(f"  CCW check: signed area of L-shape = {_area_concave:.2f} > 0  ✓")

# Convex decomposition: two axis-aligned convex pieces that tile the L-shape
# exactly without gaps or overlaps.
P_lower = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)  # lower 2×1 rectangle
P_upper = np.array([[0, 1], [1, 1], [1, 2], [0, 2]], dtype=float)  # upper 1×1 square
_decomp = [P_lower, P_upper]

# ── Evaluate composed field ──────────────────────────────────────────────────
all_pts7 = np.vstack([P_concave, P_lower, P_upper])
X7, Y7 = make_grid(all_pts7, N=N_GRID, pad_fraction=0.22)
Z7 = convex_decomp_field(X7, Y7, _decomp, delta=DELTA, n=N_ORDER)

# Verify bounded smooth union stays in [0, 1]
_z7min, _z7max = float(Z7.min()), float(Z7.max())
assert _z7min >= -1e-14, f"Composed field below 0: min={_z7min:.6e}"
assert _z7max <= 1.0 + 1e-14, f"Composed field above 1: max={_z7max:.6e}"
print(f"  Composed field range: [{_z7min:.6f}, {_z7max:.6f}]  ⊆ [0, 1]  ✓")

# ── Figure: 3-panel — contour / decomposition edges / 3-D surface ────────────
print("  Example 7: L-shape concave polygon via convex decomposition")
fig7 = plt.figure(figsize=(14, 4.8))
ax7a = fig7.add_subplot(1, 3, 1)   # filled contour with CCW boundary
ax7b = fig7.add_subplot(1, 3, 2)   # decomposition edges
ax7c = fig7.add_subplot(1, 3, 3, projection='3d')

# — Panel (a): composed field contour —
cf7 = ax7a.contourf(X7, Y7, Z7, levels=20, cmap='viridis', vmin=0, vmax=1)
plt.colorbar(cf7, ax=ax7a, label='B(x,y)')
_safe_contour(ax7a, X7, Y7, Z7, levels=[0.5], colors='white', linewidths=2)
# Draw the original CCW boundary prominently
_Pc = np.vstack([P_concave, P_concave[0]])
ax7a.plot(_Pc[:, 0], _Pc[:, 1], 'r-', linewidth=2.5, label='CCW boundary')
ax7a.plot(P_concave[:, 0], P_concave[:, 1], 'rs', markersize=7, zorder=5)
ax7a.set_aspect('equal')
ax7a.set_xlabel('x')
ax7a.set_ylabel('y')
ax7a.set_title(rf'(a) composed field  $\delta={DELTA}$')
ax7a.legend(fontsize=8, loc='upper right')

# — Panel (b): decomposition edges overlaid on the field —
ax7b.contourf(X7, Y7, Z7, levels=20, cmap='viridis', alpha=0.55, vmin=0, vmax=1)
ax7b.plot(_Pc[:, 0], _Pc[:, 1], 'r-', linewidth=2.5, label='CCW boundary', zorder=4)
for lbl, piece, ls in [('lower piece', P_lower, '--'), ('upper piece', P_upper, ':')]:
    _p = np.vstack([piece, piece[0]])
    ax7b.plot(_p[:, 0], _p[:, 1], color='royalblue', linestyle=ls,
              linewidth=1.8, label=lbl, zorder=3)
ax7b.set_aspect('equal')
ax7b.set_xlabel('x')
ax7b.set_ylabel('y')
ax7b.set_title('(b) decomposition edges')
ax7b.legend(fontsize=7, loc='upper right')

# — Panel (c): 3-D wireframe of the composed field —
ax7c.plot_wireframe(X7, Y7, Z7, rstride=6, cstride=6, color='0.35', linewidth=0.45)
ax7c.set_xlabel('x')
ax7c.set_ylabel('y')
ax7c.set_zlabel('B(x,y)')
ax7c.set_zlim(0.0, 1.05)
ax7c.view_init(elev=28, azim=-58)
ax7c.set_title('(c) surface')

fig7.suptitle(
    r'Concave polygon (L-shape, CCW) via convex decomposition'
    '\n'
    r'$B = 1-(1-B_\mathrm{lower})(1-B_\mathrm{upper})$'
    rf',  $\delta={DELTA}$,  $n={N_ORDER}$',
    fontsize=12,
)
plt.tight_layout()
fig7.canvas.manager.set_window_title("Demo — Concave polygon: convex decomposition")

# ── Figure: delta-evolution panel for the concave L-shape ────────────────────
print("  Example 7b: delta panel for concave L-shape")
_deltas7 = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
fig7b, axes7b = plt.subplots(2, 3, figsize=(13, 8.5), squeeze=False)
for _ax, _d in zip(axes7b.ravel(), _deltas7):
    _Z = convex_decomp_field(X7, Y7, _decomp, delta=_d, n=N_ORDER)
    _ax.contourf(X7, Y7, _Z, levels=20, cmap='viridis', alpha=0.75, vmin=0, vmax=1)
    _safe_contour(_ax, X7, Y7, _Z, levels=[0.5], colors='k', linewidths=1.3)
    _ax.plot(_Pc[:, 0], _Pc[:, 1], 'r-', linewidth=1.8, label='CCW boundary')
    _ax.set_aspect('equal')
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    _ax.set_title(rf'$\delta={_d}$')
fig7b.suptitle(
    r'Concave polygon (L-shape): composed field for increasing $\delta$'
    '\n'
    r'$B = 1-(1-B_\mathrm{lower})(1-B_\mathrm{upper})$',
    fontsize=12,
)
plt.tight_layout()
fig7b.canvas.manager.set_window_title("Demo — Concave polygon delta panel")

# ============================================================================
# 4. COLLECTIONS OF POLYGONS FORMING A 2D PARTITION  (Section 9)
# ============================================================================
# ── Why a raw sum is not a partition of unity ─────────────────────────────────
# For non-overlapping cells, B_k = 0 on every cell boundary (the field
# transitions from 1 to 0 over a band of width delta near each edge).  Near
# shared edges, adjacent cells each have B_k < 1, so their raw sum is < 1
# there.  The unnormalized aggregate sum is *not* a partition of unity.
#
# The normalized basis  B̂_k = B_k / max(Σ_j B_j, ε)  satisfies
# Σ_k B̂_k = 1 exactly wherever Σ_j B_j > ε, which is every interior point
# of the partition domain.
# ─────────────────────────────────────────────────────────────────────────────
print("\nCategory 4: collections of polygons forming a 2D partition")

# Irregular convex-cell partition of [0, 3] × [0, 2].
# Four convex quadrilateral cells sharing a non-axis-aligned interior boundary.
#
#  (0,2)──────────(1.5,2)──────────(3,2)
#   │                │                │
#   │  cell TL      /   cell TR       │
#   │              /                  │
#  (0,1)──────(1.2,1)──────────────(3,1)
#   │              \                  │
#   │  cell BL      \   cell BR       │
#   │                \                │
#  (0,0)──────────(1.5,0)──────────(3,0)
#
# Shared interior diagonal: (1.5,0)→(1.2,1)→(1.5,2)
partition_cells = [
    # Bottom-left trapezoid  (CCW)
    np.array([[0.0, 0.0], [1.5, 0.0], [1.2, 1.0], [0.0, 1.0]], dtype=float),
    # Bottom-right trapezoid  (CCW)
    np.array([[1.5, 0.0], [3.0, 0.0], [3.0, 1.0], [1.2, 1.0]], dtype=float),
    # Top-left trapezoid  (CCW)
    np.array([[0.0, 1.0], [1.2, 1.0], [1.5, 2.0], [0.0, 2.0]], dtype=float),
    # Top-right trapezoid  (CCW)
    np.array([[1.2, 1.0], [3.0, 1.0], [3.0, 2.0], [1.5, 2.0]], dtype=float),
]

# Verify all partition cells are CCW
for _k, _cell in enumerate(partition_cells):
    _area_k = polygon_signed_area(_cell)
    assert _area_k > 0, (
        f"Partition cell {_k} is not in CCW order (signed area = {_area_k:.4f})"
    )
print(f"  All {len(partition_cells)} partition cells verified CCW  ✓")

DELTA_PART = 0.15
_all_cell_pts = np.vstack(partition_cells)
X_part, Y_part = make_grid(_all_cell_pts, N=220, pad_fraction=0.08)

# ── Example 11: partition net + normalized-basis sum surfaces ─────────────────
print("  Example 11: partition net and normalized basis surfaces")
fig11 = partition_basis_surfaces(
    partition_cells,
    deltas=(0.05, 0.12, 0.25),
    n=N_ORDER,
    N=180,
)
fig11.suptitle(
    r"Irregular convex-cell partition: normalized basis sum  $\sum_k\hat{B}_k$"
    "\n"
    r"$\hat{B}_k = B_k\,/\,\max(\sum_j B_j,\,\varepsilon)$  —  sum $\equiv$ 1"
    r" over partition interior",
    fontsize=12,
)
plt.tight_layout()
fig11.canvas.manager.set_window_title("Demo — Normalized partition basis surfaces")

# ── Example 12: individual normalized-basis gallery ───────────────────────────
print("  Example 12: normalized basis function gallery for each cell")
_basis_gal, _raw_sum_gal = partition_basis_normalized(
    partition_cells, X_part, Y_part, delta=DELTA_PART, n=N_ORDER
)
fig12, axes12 = plt.subplots(2, 2, figsize=(10.5, 8.2), squeeze=False)
for _idx, (_ax12, _Bk) in enumerate(zip(axes12.ravel(), _basis_gal), start=1):
    cf12 = _ax12.contourf(X_part, Y_part, _Bk, levels=20, cmap='viridis',
                          vmin=0.0, vmax=1.0)
    plt.colorbar(cf12, ax=_ax12)
    _safe_contour(_ax12, X_part, Y_part, _Bk, levels=[0.5], colors='white', linewidths=1.5)
    draw_polygon_outline(partition_cells[_idx - 1], ax=_ax12, linestyle='--',
                         color='r', linewidth=1.5, marker='o', markersize=4)
    _ax12.set_aspect('equal')
    _ax12.set_xlabel('x')
    _ax12.set_ylabel('y')
    _ax12.set_title(
        rf"Normalized basis $\hat{{B}}_{_idx}$  ($\delta={DELTA_PART}$)"
    )
fig12.suptitle(
    rf"Normalized partition-basis functions  $\delta={DELTA_PART}$"
    "\n"
    r"Each $\hat{B}_k = B_k\,/\,\sum_j B_j$;  by construction $\sum_k\hat{B}_k = 1$",
    fontsize=12,
)
plt.tight_layout()
fig12.canvas.manager.set_window_title("Demo — Normalized partition basis gallery")

# ── Example 13: partition-of-unity numerical verification ────────────────────
print("  Example 13: partition-of-unity numerical check")
_basis_check, _raw_sum_check = partition_basis_normalized(
    partition_cells, X_part, Y_part, delta=DELTA_PART, n=N_ORDER
)
_Z_sum = sum(_basis_check)

# Report max |Σ B̂_k − 1| over the interior (exclude outer boundary strip)
_interior = (
    (X_part > 0.15) & (X_part < 2.85) &
    (Y_part > 0.15) & (Y_part < 1.85) &
    (_raw_sum_check > 1e-6)            # exclude points outside all cells
)
_max_err = float(np.max(np.abs(_Z_sum[_interior] - 1.0))) if _interior.any() else 0.0
_status = "✓ within tolerance" if _max_err < 1e-9 else "⚠ exceeds expected tolerance"
print(
    f"    Partition-of-unity (δ={DELTA_PART}):  "
    f"max|Σ B̂_k − 1| = {_max_err:.2e}  {_status}"
)

fig13 = plt.figure(figsize=(11, 4.5))
ax13a = fig13.add_subplot(1, 2, 1)
ax13b = fig13.add_subplot(1, 2, 2, projection='3d')

cf13 = ax13a.contourf(
    X_part, Y_part, _Z_sum, levels=np.linspace(0.8, 1.2, 25),
    cmap='RdYlGn', vmin=0.8, vmax=1.2, extend='both',
)
plt.colorbar(cf13, ax=ax13a, label=r'$\sum_k\hat{B}_k$')
for _cell in partition_cells:
    draw_polygon_outline(_cell, ax=ax13a, linestyle=':', color='0.5',
                         linewidth=0.9, marker='o', markersize=3)
ax13a.set_aspect('equal')
ax13a.set_xlabel('x')
ax13a.set_ylabel('y')
ax13a.set_title(r'Normalized sum $\sum_k\hat{B}_k$ (= 1 in interior)')

ax13b.plot_wireframe(X_part, Y_part, _Z_sum,
                     rstride=5, cstride=5, color='0.35', linewidth=0.45)
ax13b.set_xlabel('x')
ax13b.set_ylabel('y')
ax13b.set_zlabel(r'$\sum_k\hat{B}_k$')
ax13b.set_zlim(0.0, 1.05)
ax13b.view_init(elev=28, azim=-58)
ax13b.set_title(rf'Sum surface   max|sum−1|={_max_err:.1e}')

fig13.suptitle(
    rf"Partition-of-unity verification  ($\delta={DELTA_PART}$)"
    "\n"
    rf"max$|\sum_k\hat{{B}}_k - 1|$ = {_max_err:.2e} over interior",
    fontsize=12,
)
plt.tight_layout()
fig13.canvas.manager.set_window_title("Demo — Partition of unity verification")

print("\nDemo complete.")
plt.show()

