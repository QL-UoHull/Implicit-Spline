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

1. **convex polygon** demos (square/triangle/pentagon),
2. convex **delta-sweep** and **smoothness-order** (`n`) comparisons,
3. paper-style multi-panel contour evolution under varying `delta`,
4. a **freeform-like composed field** built from multiple convex components,
5. **polygons with holes** shown explicitly as composed fields
   (outer loop field multiplied by complements of convex inner-loop fields), and
6. a convex-cell **2D partition** section showing both:
   - the partition net itself, and
   - per-cell fields plus aggregate (unnormalized-sum) field surfaces.

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

## Notes

The new advanced examples are intended to better match the kinds of figures
shown in the paper:

- multi-panel contour evolution under increasing `delta`,
- freeform-like shape design via composition of convex components,
- polygon-with-hole examples, and
- 2D partition cell-wise and aggregate implicit-field surfaces.

These examples are illustrative rather than exact reproductions of any one
figure from the paper, but they provide a much closer visual and conceptual
match than the previous minimal demos.
