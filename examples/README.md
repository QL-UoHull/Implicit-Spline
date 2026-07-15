# Examples

Runnable examples are available in three forms:

## Standalone Python script (`demo.py`)

A self-contained demo that mirrors `../matlab/Demo.m`:

```bash
# from the repository root
python examples/demo.py
```

This runs four examples — unit square, equilateral triangle, regular pentagon,
and an irregular pentagon with a delta-sweep — and displays the resulting plots.

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
