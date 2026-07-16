# Examples

Runnable examples are available in three forms:

## Standalone Python script (`demo.py`)

A self-contained demo that now goes beyond the original four basic examples and
includes paper-style figures inspired by Li & Tian (2009):

```bash
# from the repository root
python examples/demo.py
```

The script includes:

1. unit square contour,
2. equilateral triangle contour,
3. regular pentagon contour,
4. delta sweep for an irregular polygon,
5. smoothness-order (`n`) sweep,
6. a six-panel paper-style contour figure showing how the implicit curve evolves
   as `delta` increases,
7. a four-panel freeform implicit-curve gallery, and
8. a polygon-partition / summed-basis surface figure in the style of the paper.

## Jupyter Notebooks (`../notebooks/`)

| Notebook | Description |
|----------|-------------|
| [`01_basic_polygon.ipynb`](../notebooks/01_basic_polygon.ipynb) | Basic usage with hard-coded polygons, delta/n sweeps |
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
- freeform curve design from hand-crafted control polygons, and
- 3D wireframe surface views of summed polygon basis functions.

These examples are illustrative rather than exact reproductions of any one
figure from the paper, but they provide a much closer visual and conceptual
match than the previous minimal demos.
