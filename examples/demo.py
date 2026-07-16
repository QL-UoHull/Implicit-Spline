"""
demo.py
-------
Standalone Python demonstration of the 2D Piecewise Algebraic Implicit Spline.

Includes both the original basic examples and paper-style visual panels inspired
by Li & Tian (2009).
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.join(_here, "..", "python")
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


def _set_window_title(fig, title):
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)


DELTA = 0.12
N_ORDER = 2
N_GRID = 200

print("=== 2D Implicit Spline Demo ===\n")

print("Example 1: unit square")
P_sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
fig1, ax1 = plt.subplots(figsize=(6, 5))
_set_window_title(fig1, "Demo — Example 1: Square")
draw_imp_spline(P_sq, delta=DELTA, n=N_ORDER, N=N_GRID, ax=ax1,
                title=rf"Example 1: Square ($\delta={DELTA}$, $n={N_ORDER}$)")
plt.tight_layout()

print("Example 2: equilateral triangle")
P_tri = np.array([[0, 0], [2, 0], [1, np.sqrt(3)]], dtype=float)
fig2, ax2 = plt.subplots(figsize=(6, 5))
_set_window_title(fig2, "Demo — Example 2: Triangle")
draw_imp_spline(P_tri, delta=DELTA, n=N_ORDER, N=N_GRID, ax=ax2,
                title=rf"Example 2: Triangle ($\delta={DELTA}$, $n={N_ORDER}$)")
plt.tight_layout()

print("Example 3: regular pentagon")
theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 6)[:-1]
P_pent = np.column_stack([np.cos(theta), np.sin(theta)])
fig3, ax3 = plt.subplots(figsize=(6, 5))
_set_window_title(fig3, "Demo — Example 3: Pentagon")
draw_imp_spline(P_pent, delta=DELTA, n=N_ORDER, N=N_GRID, ax=ax3,
                title=rf"Example 3: Pentagon ($\delta={DELTA}$, $n={N_ORDER}$)")
plt.tight_layout()

print("Example 4: sweeping delta [0.05, 0.15, 0.30]")
P_irr = np.array([[0, 0], [2, 0], [2.5, 0.8], [1.5, 1.8], [-0.2, 1.2]], dtype=float)
fig4 = compare_delta(P_irr, deltas=(0.05, 0.15, 0.30), n=N_ORDER, N=N_GRID)
_set_window_title(fig4, "Demo — Example 4: delta sweep")

print("Example 5: smoothness-order comparison [n=1,2,3]")
fig5 = compare_n(P_sq, delta=0.15, n_values=(1, 2, 3), N=N_GRID)
_set_window_title(fig5, "Demo — Example 5: smoothness comparison")

print("Example 6: paper-style six-panel delta evolution")
panel_deltas = (0.03, 0.06, 0.10, 0.16, 0.24, 0.36)
fig6 = panel_delta_shapes(
    P_irr,
    deltas=panel_deltas,
    n=2,
    N=260,
    layout=(2, 3),
    title=r"Paper-style contour evolution as $\delta$ increases",
)
_set_window_title(fig6, "Demo — Example 6: paper-style contour panels")

print("Example 7: freeform curve gallery")
gallery_polys = [
    np.array([[0.0, 0.0], [1.6, -0.2], [2.1, 0.9], [1.1, 1.8], [-0.2, 1.2]]),
    np.array([[0.0, 0.2], [0.9, -0.3], [1.8, 0.3], [1.4, 1.5], [0.3, 1.9], [-0.4, 1.0]]),
    np.array([[0.0, 0.0], [0.8, -0.4], [1.8, 0.1], [2.0, 1.0], [1.2, 1.9], [0.2, 1.6], [-0.3, 0.8]]),
    np.array([[0.0, 0.0], [1.4, -0.4], [2.4, 0.5], [1.7, 1.7], [0.6, 1.9], [-0.5, 1.1]]),
]
fig7, axes7 = plt.subplots(2, 2, figsize=(10, 8))
for i, (ax, poly) in enumerate(zip(axes7.ravel(), gallery_polys), start=1):
    draw_imp_spline(poly, delta=0.12, n=2, N=220, ax=ax, title=f"freeform {i}")
fig7.suptitle("Freeform implicit-curve gallery", fontsize=13)
plt.tight_layout()
_set_window_title(fig7, "Demo — Example 7: freeform gallery")

print("Example 8: partition and summed-basis surfaces")
partition_polys = [
    np.array([[0.0, 0.0], [1.1, 0.0], [0.95, 0.8], [0.1, 0.85]]),
    np.array([[1.1, 0.0], [2.2, 0.0], [2.0, 0.95], [0.95, 0.8]]),
    np.array([[0.1, 0.85], [0.95, 0.8], [0.9, 1.8], [0.0, 1.9]]),
    np.array([[0.95, 0.8], [2.0, 0.95], [1.9, 1.9], [0.9, 1.8]]),
]
fig8 = partition_basis_surfaces(
    partition_polys,
    deltas=(0.04, 0.10, 0.22),
    n=2,
    N=130,
)
_set_window_title(fig8, "Demo — Example 8: partition-based surface figure")

print("Example 9: extra 3D wireframe surface view")
fig9 = plt.figure(figsize=(7, 5))
ax9 = fig9.add_subplot(111, projection="3d")
draw_surface(
    P_irr,
    delta=0.12,
    n=2,
    N=100,
    ax=ax9,
    wireframe=True,
    elev=28,
    azim=-58,
    title="Wireframe view of implicit spline surface",
)
_set_window_title(fig9, "Demo — Example 9: wireframe surface")
plt.tight_layout()

print("\nDemo complete.")
plt.show()
