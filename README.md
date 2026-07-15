# 2D Piecewise Algebraic Splines for Implicit Modeling

This repository implements the method proposed by Qingde Li and Jie Tian (ACM TOG 2009) for constructing bivariate piecewise algebraic splines over arbitrary polygonal partitions.

## Key Features
- Explicit analytical formulation of spline basis functions \(B_{A,\delta}(x,y)\)
- Adjustable polygon smoothing parameter \(\delta\) controlling implicit contour proximity
- Supports **arbitrary \(C^n\) continuity** (for any prescribed integer \(n \ge 1\))
- Efficient evaluation suitable for GPU or CPU implementations
- Preserves non-negativity, partition of unity, and convex-hull–like behavior

## Mathematical Overview
Each spline basis function is defined via iterative convolution:


\[
B_{A,\delta}^{(n)}(x,y) = \big(X_A * X_\square^{(n)}\big)(x,y)
\]


where \(X_A\) is the characteristic function of polygon \(A\), and \(X_\square^{(n)}\) is an \(n\)-fold convolution of a square kernel of size \(2\delta \times 2\delta\). The resulting basis functions are piecewise polynomials with \(C^{n-1}\) continuity and can be expressed explicitly.

## Example Usage
```python
from implicit_spline import ImplicitSpline2D

# polygons: list of polygons, each as list of (x, y) vertices
spline = ImplicitSpline2D(polygons, smoothness_order=3, delta=0.1)
values = spline.evaluate(x, y)  # x, y can be scalars or grids
