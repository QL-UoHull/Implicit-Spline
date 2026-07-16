# Examples

Runnable examples are available in three forms:

## Standalone Python script (`demo.py`)

A self-contained demo organized by feature category, with paper-style figures
inspired by Li & Tian (2009):

```bash
# from the repository root
python examples/demo.py
```

The script includes:

1. **Convex polygons** — square, triangle, pentagon, delta/smoothness sweeps, and
   a paper-style multi-panel contour-evolution figure.
2. **Polygons with holes** — shown explicitly as composed implicit fields:
   outer-loop field multiplied by complements of convex inner-loop fields.
3. **Concave polygon via convex decomposition** — the correct way to represent a
   concave shape using `imp_spline_2d`:
   - One explicit **L-shaped concave polygon** with vertices listed in **CCW
     boundary order**, verified with `polygon_signed_area > 0`.
   - The shape is represented through a **convex decomposition** (two
     non-overlapping convex pieces tiling the L-shape).
   - Pieces are combined with the **bounded smooth union**
     `B = 1 − ∏_k (1 − B_k)` (`convex_decomp_field`), which stays in
     [0, 1] unlike a raw sum.
   - The original CCW boundary and the decomposition edges are drawn separately.
   - A delta-evolution panel mirrors the paper-style figures.
4. **2D polygon partition** — a valid partition-basis demonstration:
   - Four irregular convex cells with non-axis-aligned shared boundaries tiling
     [0, 3] × [0, 2].
   - **Normalized basis functions** `B̂_k = B_k / max(Σ_j B_j, ε)`
     (`partition_basis_normalized`) that satisfy partition of unity by
     construction.
   - Numerical diagnostic printed to stdout:
     `max|Σ_k B̂_k − 1| = 0.00e+00` over the partition interior.
   - Gallery of individual cell basis functions and a sum-surface verification.

### Why CCW order alone is insufficient for concave polygons

`imp_spline_2d` computes a *product* of smooth half-plane fields — one per
edge.  At a reflex vertex the interior lies on the "wrong" side of an adjacent
half-plane, so the product is 0 there even though the point is geometrically
inside the polygon.  Reordering vertices to CCW corrects the winding but
cannot repair this topological deficiency; a convex decomposition is required.

### Why a raw field sum is not a partition basis

Near shared cell boundaries, all adjacent cells have B_k < 1 (the field
transitions from 1 to 0 over a band of width δ near each edge), so their raw
sum is less than 1 there.  Normalizing each B_k by the running sum ensures
Σ_k B̂_k = 1 everywhere inside the partition domain.

## Jupyter Notebooks (`../notebooks/`)

| Notebook | Description |
|----------|-------------|
| [`01_basic_polygon.ipynb`](../notebooks/01_basic_polygon.ipynb) | Colab-visible demos for convex, concave, holes, and partition examples |
| [`02_data_from_file.ipynb`](../notebooks/02_data_from_file.ipynb) | Load polygon from `../data/sample_polygon.txt` |

## MATLAB Scripts (`../matlab/`)

| Script | Description |
|--------|-------------|
| `Demo.m` | Multi-example MATLAB demonstration |
| `Demo_v1.m` | Legacy single-polygon demo |

## Sample polygon data (`../data/`)

`sample_polygon.txt` — a six-vertex irregular hexagon in plain-text format.

