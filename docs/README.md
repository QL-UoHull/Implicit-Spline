# README - Theory and Derivations

## 2D Piecewise Algebraic Splines for Implicit Modeling

This document summarizes the mathematical framework behind **2D piecewise algebraic splines for implicit modeling** introduced by Li and Tian (ACM TOG 2009), and maps the theory to this repository’s implementation.

---

## 1) Problem setting and objective

Given a polygonal partition `{A_k}` of `R^2`, construct basis functions

`B(A_k, delta, n)(x, y)`

such that each basis is:

- piecewise polynomial and analytically expressible,
- non-negative,
- part of a partition of unity over the partition,
- locally supported (for finite polygons),
- additive on disjoint regions,
- `C^(n-1)` smooth for integer order `n >= 1`.

`delta > 0` is a smoothing scale controlling transition width near polygon boundaries.

---

## 2) Convolution-based definition

For polygon `A`, define indicator:

`X_A(x, y) = 1 if (x, y) in A, else 0`.

Let `X_square` be the indicator of the axis-aligned square centered at the origin with side length `2*delta` (support `[-delta, delta] x [-delta, delta]`).

Define order-`n` basis:

`B(A, delta, n) = X_A * X_square * ... * X_square`

where `*` is convolution and `X_square` appears `n` times.

Interpretation: repeated convolution with `X_square` smooths the hard polygon indicator into a compactly supported algebraic field.

---

## 3) Explicit algebraic construction (paper Sections 3-4)

The paper derives a closed form using a bivariate building block:

`S2(a, b, delta, n)(x, y)`

built from finite alternating sums of shifted elementary polynomials:

`A(a, b, n)(x, y)`.

High-level form:

`S2(a, b, delta, n)(x, y) = (1 / (4*delta^2)^n) * sum_{i=0..n} sum_{j=0..n} [ (-1)^(i+j) * c[i] * c[j] * F[i,j](x,y) ]`

where:
- `F[i,j]` are shifted evaluations of `A(a,b,n)`,
- `c[i]` are combinatorial coefficients from repeated box convolution.

### 3.1) Implicit geometric operators

The construction uses three objects:

1. **Implicit vertex**: translated copy of the central block at vertex `V`.
2. **Implicit edge**: oriented difference of two endpoint implicit vertices.
3. **Implicit polygon**: signed sum of implicit edges around polygon boundary.

Theorem 3.4 shows this edge-sum representation equals the convolution-defined basis `B(A, delta, n)`.

---

## 4) Derivation roadmap (implementation-oriented)

1. Geometric integral view of convolution: evaluate `B(A, delta, n)(P)` as overlap measure between polygon and translated kernel supports around point `P`.
2. Finite-difference pattern from repeated box convolution: for `n=1,2,3`, masks appear as `2x2`, `3x3`, `4x4`.
3. Boundary decomposition: convert area expression to oriented boundary terms (edge contributions as endpoint differences).
4. Special/degenerate cases: axis-aligned and vertical/horizontal edges are handled explicitly (including smooth-step `H_n` formulations).

---

## 5) Fundamental properties

From convolution and explicit formulas:

- Boundedness and non-negativity:  
  `0 <= B(A, delta, n)(x, y) <= 1`.

- Partition of unity: if `{A_k}` partitions `R^2` up to measure-zero overlaps, then  
  `sum_k B(A_k, delta, n)(x, y) = 1`.

- Local support: finite polygons induce finite support neighborhoods.

- Additivity on disjoint sets: if `A intersect C = empty`, then  
  `B(A union C, delta, n) = B(A, delta, n) + B(C, delta, n)`.

- Smoothness: `B(A, delta, n)` is `C^(n-1)`.

---

## 6) Parameter effects and practical guidance

### 6.1) Smoothness order `n`
- Larger `n`: smoother fields, broader transition, higher polynomial degree/cost.
- Smaller `n`: sharper transition, lower cost, lower differentiability.

### 6.2) Smoothing scale `delta`
- Smaller `delta`: contours track polygon boundaries more tightly.
- Larger `delta`: stronger smoothing and wider blending zones.

Rule of thumb: choose `delta` as a fraction of the smallest local feature size; reduce sampling step as `delta` decreases.

### 6.3) Contour isovalue
For normalized occupancy-like fields, `tau = 0.5` is common.

### 6.4) Vertex orientation
Use consistent polygon orientation (typically CCW for outer boundaries). If holes are represented, orient inner boundaries oppositely.

---

## 7) Code mapping in this repository

Primary implementation:

- `src/implicit_spline.py`

Expected responsibilities:

- evaluate `A(a,b,n)`,
- assemble `S2(a,b,delta,n)` via finite shifted masks,
- construct implicit vertex/edge/polygon contributions,
- compute weighted sums and contour extraction (including forms related to Eq. (5) and Eq. (20) in the paper).

Suggested public parameters:

- `smoothness_order` (n),
- `delta`,
- polygon winding/orientation,
- contour `isovalue`,
- sampling resolution/grid step.

---

## 8) Numerical and robustness notes

- Guard near-zero edge lengths and nearly collinear triples.
- Verify partition-of-unity numerically on test partitions.
- Regression tests:
  - single rectangle,
  - adjacent polygons (continuity and sum-to-one),
  - parameter sweep over `n` and `delta`.
- Performance: precompute mask coefficients and reuse shifted evaluations.

---

## 9) Minimal validation checklist

1. `0 <= B <= 1` on sampled domain.
2. `sum_k B_k ~= 1` on partition test set.
3. No visible seams across shared edges.
4. Sharper boundary behavior as `delta -> 0+` (with refined sampling).
5. Observed continuity order matches `n`.

---

## Reference

Li, Q. and Tian, J. (2009).  
**2D Piecewise Algebraic Splines for Implicit Modeling.**  
ACM Transactions on Graphics, 28(2), Article 13.  
DOI: https://doi.org/10.1145/1516522.1516524
