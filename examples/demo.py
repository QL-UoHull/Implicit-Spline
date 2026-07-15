"""
demo.py
-------
Standalone Python demonstration of the 2D Piecewise Algebraic Implicit Spline.

Mirrors matlab/Demo.m — runs four examples that showcase the main features
of the Li & Tian (2009) implicit-spline construction:

  Example 1 : Unit square (right-angle corners)
  Example 2 : Equilateral triangle (acute corners)
  Example 3 : Regular pentagon
  Example 4 : Irregular pentagon — sweeping delta to show bandwidth effect

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
from implicit_spline.visualization import draw_surface, compare_delta

# ── Shared parameters ─────────────────────────────────────────────────────────
DELTA = 0.12   # transition bandwidth
N_ORDER = 2    # smoothness order (C^n near each edge)
N_GRID = 200   # grid resolution (N × N points)

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

# ── Example 3: Regular pentagon ───────────────────────────────────────────────
print("Example 3: regular pentagon")
theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 6)[:-1]  # 5 vertices, start at top
P_pent = np.column_stack([np.cos(theta), np.sin(theta)])

fig3, ax3 = plt.subplots(figsize=(6, 5))
fig3.canvas.manager.set_window_title("Demo — Example 3: Pentagon")
draw_imp_spline(P_pent, delta=DELTA, n=N_ORDER, N=N_GRID, ax=ax3,
                title=rf"Example 3: Pentagon ($\delta={DELTA}$, $n={N_ORDER}$)")
plt.tight_layout()

# ── Example 4: Effect of delta (bandwidth) ────────────────────────────────────
print("Example 4: sweeping delta  [0.05, 0.15, 0.30]")
P_irr = np.array([[0, 0], [2, 0], [2.5, 0.8], [1.5, 1.8], [-0.2, 1.2]], dtype=float)

fig4 = compare_delta(P_irr, deltas=(0.05, 0.15, 0.30), n=N_ORDER, N=N_GRID)
fig4.canvas.manager.set_window_title("Demo — Example 4: delta sweep")

print("\nDemo complete.")
plt.show()
