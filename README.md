# 2D Piecewise Algebraic Splines for Implicit Modeling

This repository implements the method proposed by Qingde Li and Jie Tian (ACM TOG 2009) for constructing bivariate piecewise algebraic splines over arbitrary polygonal partitions.

## Key Features
- Explicit analytical formulation of spline basis functions \(B_{A,\delta}(x,y)\)
- Adjustable polygon smoothing parameter δ controlling implicit contour proximity
- Supports **arbitrary $\(C^n\)$ continuity** (for any prescribed integer $\(n \ge 1\))$
- Efficient evaluation suitable for GPU or CPU implementations
- Preserves non-negativity, partition of unity, and convex-hull–like

## Mathematical Overview
Each spline basis function is defined via iterative convolution:


\[
B_{A,\delta}(x,y) = \int_{\mathbb{R}^2} X_A(s,t) X_\square(s-x,t-y)\,ds\,dt
\]


where \(X_A\) is the characteristic function of polygon A and \(X_\square\) represents a square kernel of size 2δ×2δ.

## Example Usage
```python
from implicit_spline import ImplicitSpline2D
spline = ImplicitSpline2D(polygons, smoothness=2, delta=0.1)
values = spline.evaluate(x, y)
