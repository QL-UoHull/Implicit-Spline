# 2D Piecewise Algebraic Splines for Implicit Modeling

This repository disseminates the paper and hosts a **reference implementation**:

**Qingde Li** and **Jie Tian**, *2D Piecewise Algebraic Splines for Implicit Modeling*, ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH 2009).

[![Open In Colab – basic demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QL-UoHull/Implicit-Spline/blob/main/notebooks/01_basic_polygon.ipynb)

## Authors and affiliations
- Qingde Li — University of Hull, United Kingdom
- Jie Tian — Chinese Academy of Sciences, China

## Key Features
- Explicit analytical formulation of spline basis functions $B_{A,\delta}(x,y)$
- Adjustable polygon smoothing parameter $\delta$ controlling implicit contour proximity
- Supports **arbitrary $C^n$ continuity** (for any prescribed integer $n \ge 1$)
- Efficient evaluation suitable for GPU or CPU implementations
- Preserves non-negativity, partition of unity, and convex-hull–like behavior

## Repository status
✅ **Status:** This repository now includes a **reference implementation** in both MATLAB and Python, together with Jupyter notebook demos.

## Repository structure

```text
.
├── CITATION.cff                       # Citation metadata
├── LICENSE                            # MIT licence
├── README.md                          # This file
├── requirements.txt                   # Python dependencies (numpy, matplotlib)
│
├── matlab/                            # MATLAB reference implementation
│   ├── H.m                            # Smooth Heaviside step function
│   ├── Lxy.m                          # Signed distance to directed line
│   ├── Lxy00.m                        # Unnormalised signed linear function
│   ├── Point_imp.m                    # Squared distance to point
│   ├── LineSeg_imp.m                  # Smooth segment implicit function
│   ├── L_corner_inter.m               # General corner blending function
│   ├── Square_Angle_inter.m           # Right-angle corner variant
│   ├── U_Angle_inter.m                # Obtuse/reflex corner variant
│   ├── ImpSpline2D.m                  # Main 2D implicit spline assembler
│   ├── DrawImpSpline.m                # Grid evaluation + visualisation
│   ├── DrawImpSpline_V1.m             # Legacy simplified visualisation
│   ├── dataFrGr.m                     # Interactive polygon input (ginput)
│   ├── Demo.m                         # Main demo script
│   └── Demo_v1.m                      # Legacy demo
│
├── python/
│   └── implicit_spline/               # Python package
│       ├── __init__.py
│       ├── core.py                    # H, lxy, imp_spline_2d, …
│       └── visualization.py           # draw_imp_spline, draw_surface, …
│
├── notebooks/
│   ├── 01_basic_polygon.ipynb         # Hard-coded polygon demo (Colab-ready)
│   └── 02_data_from_file.ipynb        # Load vertices from file + plots
│
├── data/
│   └── sample_polygon.txt             # Example polygon vertices (text)
│
├── docs/
│   └── README.md
└── examples/
    └── README.md
```

---

## Running the MATLAB implementation

1. Open MATLAB and `cd` into the `matlab/` directory:

   ```matlab
   cd path/to/Implicit-Spline/matlab
   ```

2. Run the main demo:

   ```matlab
   Demo
   ```

3. To evaluate the spline for your own polygon:

   ```matlab
   P = [0 0; 1 0; 1 1; 0 1];          % CCW or CW — auto-corrected
   delta = 0.12;                        % transition bandwidth
   n = 2;                               % C^n smoothness order
   DrawImpSpline(P, delta, n, 200);     % evaluate + plot
   ```

4. For interactive vertex input:

   ```matlab
   P = dataFrGr();                      % click vertices; right-click to finish
   DrawImpSpline(P, 0.12, 2, 200);
   ```

### MATLAB requirements
- MATLAB R2016b or later (uses `circshift`, `nchoosek`, dot-operator broadcasting).
- No additional toolboxes required.

---

## Running the Python implementation

### Install dependencies

```bash
pip install -r requirements.txt
```

### Quick start

```python
import numpy as np
from implicit_spline import imp_spline_2d, draw_imp_spline

P = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
draw_imp_spline(P, delta=0.12, n=2, N=300)
```

### Using from the repository root (no install)

```python
import sys
sys.path.insert(0, 'python')
from implicit_spline import imp_spline_2d
```

---

## Jupyter / Colab notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| [`01_basic_polygon.ipynb`](notebooks/01_basic_polygon.ipynb) | H function, multiple polygons, δ/n sweeps | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QL-UoHull/Implicit-Spline/blob/main/notebooks/01_basic_polygon.ipynb) |
| [`02_data_from_file.ipynb`](notebooks/02_data_from_file.ipynb) | Load from `data/sample_polygon.txt`, gradient plots | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QL-UoHull/Implicit-Spline/blob/main/notebooks/02_data_from_file.ipynb) |

Run locally:

```bash
pip install jupyter
jupyter notebook notebooks/
```

---

## Expected outputs

Running `Demo.m` or notebook `01` produces:

- A **filled contour plot** of f(x,y) (blue = 0, yellow = 1).
- A **white iso-curve** at f = 0.5, which closely traces the polygon boundary.
- A **surface plot** showing the smooth tent-shaped function.

Increasing `delta` widens the transition zone; increasing `n` raises the
smoothness order (C^n continuity near each edge).

---

## Known limitations

- The product-of-step-functions construction is **exact for convex polygons**.
  For non-convex (concave) polygons, the interior value near reflex vertices
  may be reduced; increase `delta` or decompose the polygon into convex parts.
- The MATLAB interactive demo (`dataFrGr`) requires a display; it cannot run
  in headless / batch mode.
- Very small `delta` values relative to the polygon size can cause numerical
  underflow in the product accumulator for high vertex counts.

## Usage notes

## Citation
If this work is relevant to your research, please cite the paper using the metadata in [`CITATION.cff`](./CITATION.cff).

BibTeX:

```bibtex
@article{Li2009Piecewise,
  author    = {Li, Qingde and Tian, Jie},
  title     = {2D Piecewise Algebraic Splines for Implicit Modeling},
  journal   = {ACM Transactions on Graphics},
  year      = {2009},
  volume    = {28},
  number    = {3},
  doi       = {10.1145/1516522.1516524}
}
```

## License
This repository is released under the MIT License (see [`LICENSE`](./LICENSE)).

## Contributing
Contributions are welcome, especially around documentation, reproducible examples, and implementation packaging. Please open an issue first to discuss substantial changes.
