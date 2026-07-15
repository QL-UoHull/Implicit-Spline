# Data

## `sample_polygon.txt`

A six-vertex irregular hexagon stored as plain text, one vertex per line
(format: `x  y` with space separator, comment lines start with `#`).

Used by `notebooks/02_data_from_file.ipynb` and the MATLAB `Demo.m`.

### Format for custom polygon files

```
# Optional comment lines (lines starting with #)
x0 y0
x1 y1
...
xm-1 ym-1
```

Load in Python:
```python
import numpy as np
P = np.loadtxt('data/my_polygon.txt', comments='#')
```

Load in MATLAB:
```matlab
P = load('data/my_polygon.txt');
```
