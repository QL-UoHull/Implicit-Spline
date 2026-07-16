"""
visualization.py
----------------
Matplotlib-based helpers for evaluating and displaying 2D implicit splines.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .core import imp_spline_2d, partition_basis_normalized


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_contour(ax, X, Y, Z, levels, **kwargs):
    """Draw contours only for levels that lie strictly within the data range.

    Avoids the Matplotlib ``ValueError`` / ``UserWarning`` that occurs when
    all requested levels fall outside ``[Z.min(), Z.max()]``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    X, Y, Z : ndarray
    levels : sequence of float
    **kwargs
        Forwarded to ``ax.contour``.
    """
    zmin, zmax = float(Z.min()), float(Z.max())
    valid = [lvl for lvl in levels if zmin < lvl < zmax]
    if valid:
        ax.contour(X, Y, Z, levels=valid, **kwargs)


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


def _close_polygon(P):
    """Return *P* with its first vertex appended to close the loop."""
    P = np.asarray(P, dtype=float)
    return np.vstack([P, P[0]])


def draw_polygon_outline(
    P,
    ax=None,
    *,
    linestyle: str = ":",
    color: str = "0.5",
    linewidth: float = 0.8,
    marker: str = "o",
    markersize: float = 2.5,
    annotate_vertices: bool = False,
    vertex_labels=None,
    vertex_text_offset=(0.02, 0.02),
    title: str | None = None,
):
    """Draw a polygon outline and optional vertex markers/labels.

    Parameters
    ----------
    P : array-like, shape (m, 2)
        Polygon vertices.
    ax : matplotlib.axes.Axes, optional
        Target axes.  A new figure is created if *None*.
    linestyle, color, linewidth, marker, markersize : optional
        Styling options passed to Matplotlib.
    annotate_vertices : bool
        If True, annotate each vertex.
    vertex_labels : sequence of str, optional
        Custom labels for each vertex.  Defaults to ``P0, P1, ...``.
    vertex_text_offset : tuple[float, float]
        Offset added to each label position.
    title : str, optional
        Axes title.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes used for plotting.
    """
    P = np.asarray(P, dtype=float)
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(5, 4))

    P_closed = _close_polygon(P)
    ax.plot(
        P_closed[:, 0],
        P_closed[:, 1],
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        P[:, 0],
        P[:, 1],
        linestyle="None",
        marker=marker,
        color=color,
        markersize=markersize,
    )

    if annotate_vertices:
        if vertex_labels is None:
            vertex_labels = [f"P{i}" for i in range(len(P))]
        dx, dy = vertex_text_offset
        for (xv, yv), label in zip(P, vertex_labels):
            ax.text(xv + dx, yv + dy, str(label), fontsize=8, color=color)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)

    if created_fig:
        plt.tight_layout()

    return ax


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
    _safe_contour(ax, X, Y, Z, levels=[iso_level], colors="white", linewidths=2)

    if show_polygon:
        P_closed = _close_polygon(P)
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
    *,
    wireframe: bool = False,
    elev: float | None = None,
    azim: float | None = None,
    zlim=None,
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
    wireframe : bool, optional
        If True, draw a wireframe instead of a shaded surface.
    elev, azim : float, optional
        3D view angles passed to ``ax.view_init``.
    zlim : tuple[float, float], optional
        Explicit z-axis limits.

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

    if wireframe:
        ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, color="0.35", linewidth=0.5)
    else:
        ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            alpha=0.9,
        )

    if elev is not None or azim is not None:
        ax.view_init(
            elev=elev if elev is not None else ax.elev,
            azim=azim if azim is not None else ax.azim,
        )
    if zlim is not None:
        ax.set_zlim(*zlim)

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


def compare_n(
    P,
    delta: float = 0.15,
    n_values=(1, 2, 3),
    N: int = 200,
    iso_level: float = 0.5,
    cmap: str = "viridis",
):
    """Compare the same polygon for several smoothness orders *n*."""
    fig, axes = plt.subplots(1, len(n_values), figsize=(5 * len(n_values), 4), squeeze=False)
    axes = axes[0]

    for ax, nn in zip(axes, n_values):
        draw_imp_spline(P, delta=delta, n=nn, N=N, ax=ax, iso_level=iso_level, cmap=cmap)
        ax.set_title(rf"$n = {nn}$")

    fig.suptitle(rf"Effect of smoothness order $n$ ($\delta={delta}$)", fontsize=13)
    plt.tight_layout()
    return fig


def panel_delta_shapes(
    P,
    deltas,
    *,
    n: int = 2,
    N: int = 300,
    iso_level: float = 0.5,
    layout=(2, 3),
    title: str | None = None,
    show_vertices: bool = True,
    vertex_labels=None,
):
    """Draw paper-style contour panels for a sequence of ``delta`` values."""
    rows, cols = layout
    if rows * cols < len(deltas):
        raise ValueError("layout does not provide enough subplots for all delta values")

    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.6 * rows), squeeze=False)
    axes_flat = axes.ravel()

    P = np.asarray(P, dtype=float)
    X, Y = make_grid(P, N=N, pad_fraction=0.25)

    for ax, d in zip(axes_flat, deltas):
        Z = imp_spline_2d(X, Y, P, delta=d, n=n)
        _safe_contour(ax, X, Y, Z, levels=[iso_level], colors="k", linewidths=0.85)
        draw_polygon_outline(
            P,
            ax=ax,
            linestyle=":",
            color="0.65",
            linewidth=0.8,
            marker="*" if show_vertices else "None",
            markersize=5.5 if show_vertices else 0,
            annotate_vertices=False,
        )
        if vertex_labels is not None:
            dx, dy = 0.025, 0.025
            for (xv, yv), label in zip(P, vertex_labels):
                ax.text(xv + dx, yv + dy, str(label), fontsize=7, color="0.25")
        ax.set_title(rf"$\delta$={d}")
        ax.grid(False)

    for ax in axes_flat[len(deltas):]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


def partition_basis_surfaces(
    polygons,
    deltas=(0.05, 0.10, 0.20),
    *,
    n: int = 2,
    N: int = 140,
    figsize=(11, 8),
):
    """Create a paper-style 2×2 figure for a polygon partition and normalized-basis sums.

    The first panel shows the partition net.  The remaining three panels show
    the **normalized** sum :math:`\\sum_k \\hat{B}_k` of per-cell basis
    functions for different smoothing parameters.  Because the basis is
    normalized, the sum equals 1 everywhere inside the partition domain.

    Parameters
    ----------
    polygons : list of array-like, each shape (m_k, 2)
        Non-overlapping convex polygon cells of the partition.
    deltas : tuple of float, length 3
        Transition bandwidths to show (one 3-D panel each).
    n : int
        Smoothness order.  Default: 2.
    N : int
        Grid resolution.  Default: 140.
    figsize : tuple
        Figure size.  Default: (11, 8).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    polygons = [np.asarray(P, dtype=float) for P in polygons]
    if len(deltas) != 3:
        raise ValueError("deltas must contain exactly three values for a 2x2 panel")

    fig = plt.figure(figsize=figsize)
    ax_net = fig.add_subplot(2, 2, 1)
    ax_surfaces = [
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]

    for poly in polygons:
        draw_polygon_outline(poly, ax=ax_net, linestyle="-", color="0.2", linewidth=0.8,
                             marker="None", markersize=0)
    ax_net.set_title("(a) polygon partition net")

    all_pts = np.vstack(polygons)
    X, Y = make_grid(all_pts, N=N, pad_fraction=0.10)

    for idx, (ax, d) in enumerate(zip(ax_surfaces, deltas), start=1):
        basis, raw_sum = partition_basis_normalized(polygons, X, Y, delta=d, n=n)
        Z = sum(basis)
        ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, color="0.35", linewidth=0.45)
        ax.view_init(elev=28, azim=-58)
        ax.set_zlim(0.0, 1.05)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(r"$\sum_k\hat{B}_k$")
        panel_label = chr(ord("a") + idx)
        ax.set_title(rf"({panel_label}) $\delta={d}$  [normalized sum]")

    plt.tight_layout()
    return fig
