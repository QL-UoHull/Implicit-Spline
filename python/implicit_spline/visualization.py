"""
visualization.py
----------------
Matplotlib-based helpers for evaluating and displaying 2D implicit splines.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .core import imp_spline_2d


def make_grid(P, N: int = 200, pad_fraction: float = 0.2):
    """Build a uniform evaluation grid around polygon *P*.

    Parameters
    ----------
    P : array-like, shape (m, 2)
        Polygon vertices.
    N : int
        Number of grid points along each axis.  Default: 200.
    pad_fraction : float
        Fraction of the bounding-box span to add as padding on each side.
        Default: 0.2 (20 %).

    Returns
    -------
    X, Y : ndarray, shape (N, N)
        Meshgrid coordinate arrays.
    """
    P = np.asarray(P, dtype=float)
    x_span = P[:, 0].max() - P[:, 0].min()
    y_span = P[:, 1].max() - P[:, 1].min()
    pad = pad_fraction * max(x_span, y_span, 1e-6)

    x_lin = np.linspace(P[:, 0].min() - pad, P[:, 0].max() + pad, N)
    y_lin = np.linspace(P[:, 1].min() - pad, P[:, 1].max() + pad, N)
    return np.meshgrid(x_lin, y_lin)


def draw_imp_spline(
    P,
    delta: float = 0.1,
    n: int = 2,
    N: int = 200,
    ax=None,
    iso_level: float = 0.5,
    show_polygon: bool = True,
    cmap: str = "viridis",
    title: str | None = None,
):
    """Evaluate the implicit spline on a grid and render a contour plot.

    Parameters
    ----------
    P : array-like, shape (m, 2)
        Polygon vertices.
    delta : float
        Transition bandwidth.  Default: 0.1.
    n : int
        Smoothness order.  Default: 2.
    N : int
        Grid resolution (N × N points).  Default: 200.
    ax : matplotlib.axes.Axes, optional
        Target axes.  A new figure is created if *None*.
    iso_level : float
        Contour level to highlight (typically 0.5).  Default: 0.5.
    show_polygon : bool
        Overlay the original polygon boundary.  Default: True.
    cmap : str
        Matplotlib colormap name.  Default: ``'viridis'``.
    title : str, optional
        Axes title.  Auto-generated from parameters if *None*.

    Returns
    -------
    Z : ndarray, shape (N, N)
        Evaluated function values.
    X, Y : ndarray, shape (N, N)
        Grid coordinates.
    ax : matplotlib.axes.Axes
        The axes used for plotting.
    """
    P = np.asarray(P, dtype=float)
    X, Y = make_grid(P, N=N)
    Z = imp_spline_2d(X, Y, P, delta=delta, n=n)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(6, 5))

    cf = ax.contourf(X, Y, Z, levels=20, cmap=cmap)
    plt.colorbar(cf, ax=ax)
    ax.contour(X, Y, Z, levels=[iso_level], colors="white", linewidths=2)

    if show_polygon:
        P_closed = np.vstack([P, P[0]])
        ax.plot(P_closed[:, 0], P_closed[:, 1], "r--", linewidth=1.5, label="Polygon")
        ax.plot(P[:, 0], P[:, 1], "r.", markersize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        title if title is not None
        else rf"Implicit Spline ($\delta={delta}$, $n={n}$)"
    )

    if created_fig:
        plt.tight_layout()

    return Z, X, Y, ax


def draw_surface(
    P,
    delta: float = 0.1,
    n: int = 2,
    N: int = 100,
    ax=None,
    cmap: str = "plasma",
    title: str | None = None,
):
    """Render a 3D surface plot of the implicit spline function.

    Parameters
    ----------
    P : array-like, shape (m, 2)
        Polygon vertices.
    delta : float
        Transition bandwidth.  Default: 0.1.
    n : int
        Smoothness order.  Default: 2.
    N : int
        Grid resolution.  Default: 100.
    ax : matplotlib axes (3D), optional
        A 3D axes; created via ``projection='3d'`` if *None*.
    cmap : str
        Colormap.  Default: ``'plasma'``.
    title : str, optional
        Axes title.

    Returns
    -------
    Z : ndarray
        Evaluated function values.
    X, Y : ndarray
        Grid coordinates.
    ax : matplotlib 3D axes
    """
    P = np.asarray(P, dtype=float)
    X, Y = make_grid(P, N=N)
    Z = imp_spline_2d(X, Y, P, delta=delta, n=n)

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap,
                    linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_title(
        title if title is not None
        else rf"Surface: $\delta={delta}$, $n={n}$"
    )

    if created_fig:
        plt.tight_layout()

    return Z, X, Y, ax


def compare_delta(
    P,
    deltas=(0.05, 0.15, 0.30),
    n: int = 2,
    N: int = 200,
    iso_level: float = 0.5,
    cmap: str = "viridis",
):
    """Side-by-side comparison of the implicit spline for multiple delta values.

    Parameters
    ----------
    P : array-like, shape (m, 2)
        Polygon vertices.
    deltas : sequence of float
        Transition bandwidths to compare.  Default: (0.05, 0.15, 0.30).
    n : int
        Smoothness order.  Default: 2.
    N : int
        Grid resolution.  Default: 200.
    iso_level : float
        Contour level to highlight.  Default: 0.5.
    cmap : str
        Colormap.  Default: ``'viridis'``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    nd = len(deltas)
    fig, axes = plt.subplots(1, nd, figsize=(5 * nd, 4), squeeze=False)
    axes = axes[0]

    for ax, d in zip(axes, deltas):
        draw_imp_spline(P, delta=d, n=n, N=N, ax=ax, iso_level=iso_level, cmap=cmap)
        ax.set_title(rf"$\delta = {d}$")

    fig.suptitle("Effect of transition bandwidth $\\delta$", fontsize=13)
    plt.tight_layout()
    return fig
