"""
demo.py
-------
Standalone Python demonstration of the 2D Piecewise Algebraic Implicit Spline.

Extends the original basic examples with paper-style figures inspired by
Li & Tian (2009):

  Example 1 : Unit square (right-angle corners)
  Example 2 : Equilateral triangle (acute corners)
  Example 3 : Regular pentagon
  Example 4 : Irregular pentagon — sweeping delta to show bandwidth effect
  Example 5 : Effect of smoothness order n
  Example 6 : Paper-style multi-panel delta figure
  Example 7 : Freeform implicit curve gallery
  Example 8 : Polygon partition and summed basis surfaces

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

from implicit_spline import draw_imp_spline
from implicit_spline.visualization import (
    compare_delta,
    compare_n,
    draw_surface,
    panel_delta_shapes,
    partition_basis_surfaces,
)

# ── Shared parameters ─────────────────────────────────────────────────────────
DELTA = 0.12   # transition bandwidth
N_ORDER = 2    # smoothness order (C^n near each edge)
N_GRID = 240   # grid resolution (N × N points)

print("=== 2D Implicit Spline Demo ===\n")

# ── Example 1: Unit square ────────────────────────────────────────────────────
print("Example 1: unit square")
P_sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

fig1, ax1 = plt.subplots(figsize=(6, 5))
fig1.canvas.manager.set_window_title("Demo — Example 1: Square")
draw_imp_spline(P_sq, delta=DELTA, n=N_ORDER, N=N_GRID, ax=ax1,
                title=rf"Example 1: Square ($\delta={DELTA}$, $n={N_ORDER}$)")
plt.tight_layout()

# ── Example 2: Equilateral triangle ──────────────────────────────────────────
print("Example 2: equilateral triangle")
P_tri = np.array([[0, 0], [2, 0], [1, np.sqrt(3)]], dtype=float)

fig2, ax2 = plt.subplots(figsize=(6, 5))
fig2.canvas.manager.set_window_title("Demo — Example 2: Triangle")
draw_imp_spline(P_tri, delta=DELTA, n=N_ORDER, N=N_GRID, ax=ax2,
                title=rf"Example 2: Triangle ($\delta={DELTA}$, $n={N_ORDER}$)")
plt.tight_layout()

# ── Example 3: Regular pentagon ──────────────────────────────────────────────
print("Example 3: regular pentagon")
theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 6)[:-1]
P_pent = np.column_stack([np.cos(theta), np.sin(theta)])

fig3, ax3 = plt.subplots(figsize=(6, 5))
fig3.canvas.manager.set_window_title("Demo — Example 3: Pentagon")
draw_imp_spline(P_pent, delta=DELTA, n=N_ORDER, N=N_GRID, ax=ax3,
                title=rf"Example 3: Pentagon ($\delta={DELTA}$, $n={N_ORDER}$)")
plt.tight_layout()

# ── Example 4: Effect of delta (bandwidth) ───────────────────────────────────
print("Example 4: sweeping delta  [0.05, 0.15, 0.30]")
P_irr = np.array([[0, 0], [2, 0], [2.5, 0.8], [1.5, 1.8], [-0.2, 1.2]], dtype=float)

fig4 = compare_delta(P_irr, deltas=(0.05, 0.15, 0.30), n=N_ORDER, N=N_GRID)
fig4.canvas.manager.set_window_title("Demo — Example 4: delta sweep")

# ── Example 5: Effect of smoothness order ────────────────────────────────────
print("Example 5: sweeping n  [1, 2, 3]")
fig5 = compare_n(P_sq, delta=0.15, n_values=(1, 2, 3), N=N_GRID)
fig5.canvas.manager.set_window_title("Demo — Example 5: smoothness order")

# ── Example 6: Paper-style multi-panel delta figure ──────────────────────────
print("Example 6: paper-style delta panel")
P_ctrl = np.array([
    [-0.90,  0.80],
    [-0.08,  0.40],
    [ 0.72,  0.84],
    [ 1.02, -0.80],
    [ 0.25, -0.45],
    [ 0.00, -0.60],
    [-0.38, -0.58],
    [-0.92, -0.76],
    [-0.48,  0.00],
], dtype=float)
fig6 = panel_delta_shapes(
    P_ctrl,
    deltas=(0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
    n=2,
    N=320,
    layout=(2, 3),
    title="Paper-style contour evolution under increasing $\\delta$",
)
fig6.canvas.manager.set_window_title("Demo — Example 6: paper-style delta panel")

# ── Example 7: Freeform implicit curve gallery ───────────────────────────────
print("Example 7: freeform implicit curve gallery")
freeform_polygons = [
    np.array([[-1.0, -0.9], [-0.2, 0.72], [0.0, -0.05], [0.35, 0.78], [1.0, -0.82]]),
    np.array([[-1.0, 0.9], [1.1, 0.82], [0.78, 0.45], [-0.72, 0.10], [0.88, -0.52], [1.02, -0.92], [-0.92, -0.78]]),
    np.array([[-0.95, -0.10], [-0.55, 0.78], [0.00, 0.98], [0.55, 0.80], [0.95, -0.08], [0.60, -0.88], [-0.55, -0.78]]),
    np.array([[-0.90,  0.95], [0.88, 0.92], [0.82, 0.28], [0.12, 0.22], [0.72, -0.12], [0.66, -0.92], [-0.86, -0.92], [-0.82, -0.18], [-0.06, -0.18], [-0.68, 0.16], [-0.82, 0.62]]),
]
fig7, axes7 = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
for idx, (ax, poly) in enumerate(zip(axes7.ravel(), freeform_polygons), start=1):
    draw_imp_spline(poly, delta=0.18, n=2, N=320, ax=ax, title=f"Freeform design {idx}")
fig7.suptitle("Freeform implicit curve designs inspired by the paper", fontsize=13)
plt.tight_layout()
fig7.canvas.manager.set_window_title("Demo — Example 7: freeform gallery")

# ── Example 8: Polygon partition and summed basis surfaces ───────────────────
print("Example 8: polygon partition and summed basis surfaces")
partition_polygons = [
    np.array([[-0.95, -0.42], [-0.55,  0.10], [-0.55, -1.00]]),
    np.array([[-0.95,  0.45], [-0.55,  0.10], [-0.55,  0.85], [-0.35, 0.98], [-0.02, 0.62], [-0.32, 0.12]]),
    np.array([[-0.55,  0.10], [-0.02, 0.62], [0.48, 0.78], [0.72, 0.55], [0.48, 0.12], [0.12, -0.02], [-0.32, 0.12]]),
    np.array([[-0.32, 0.12], [0.12, -0.02], [0.35, -0.42], [-0.15, -0.62], [-0.55, -0.55], [-0.55, 0.10]]),
    np.array([[0.12, -0.02], [0.48, 0.12], [0.70, -0.30], [0.35, -0.42]]),
    np.array([[0.48, 0.12], [0.72, 0.55], [1.05, 0.42], [0.85, -0.18], [0.70, -0.30]]),
    np.array([[0.35, -0.42], [0.70, -0.30], [0.85, -0.18], [0.62, -0.92], [0.18, -0.98], [-0.15, -0.62]]),
]
fig8 = partition_basis_surfaces(partition_polygons, deltas=(0.05, 0.10, 0.20), n=2, N=150)
fig8.canvas.manager.set_window_title("Demo — Example 8: partition surfaces")

# ── Additional standalone surface ────────────────────────────────────────────
print("Additional surface view: irregular pentagon")
fig9 = plt.figure(figsize=(7, 5))
ax9 = fig9.add_subplot(111, projection='3d')
draw_surface(P_irr, delta=0.15, n=2, N=120, ax=ax9,
             title=r"Irregular pentagon surface ($\delta=0.15$, $n=2$)",
             wireframe=True, elev=28, azim=-58, zlim=(0.0, 1.1))
fig9.canvas.manager.set_window_title("Demo — Additional surface")

print("\nDemo complete.")
plt.show()
