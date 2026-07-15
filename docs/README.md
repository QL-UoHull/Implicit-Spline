
# README — Theory and Derivations

## 2D Piecewise Algebraic Splines for Implicit Modeling

This document summarizes the mathematical framework behind **2D piecewise algebraic splines for implicit modeling** introduced by Li & Tian (ACM TOG 2009), and maps the theory to this repository’s implementation.

It is intended for:
- researchers who want a compact derivation roadmap,
- implementers who need formula-to-code correspondence,
- users tuning parameters (`n`, `delta`, contour isovalue) for practical modeling.

---

## 1) Problem setting and objective

Given a polygonal partition `{A_k}` of `R^2`, construct basis functions `B_{A_k,delta}^{(n)}(x,y)` such that each basis is:

- **piecewise polynomial** and analytically expressible,
- **non-negative**,
- part of a **partition of unity** over the partition,
- **locally supported** (for finite polygons),
- **additive** on disjoint regions,
- `C^(n-1)`-smooth for chosen integer order `n >= 1`.

Here, `delta > 0` is a geometric smoothing scale controlling transition width near polygon boundaries.

---

## 2) Convolution-based definition

For polygon `A`, define its indicator:

`X_A(x,y) = 1 if (x,y) in A, else 0`.

Let `X_square` be the indicator of the axis-aligned square centered at the origin with side length `2*delta`, i.e. support `[-delta,delta] x [-delta,delta]`.

The order-`n` implicit spline basis is:

`B_{A,delta}^{(n)} = X_A * X_square * ... * X_square`  (n convolutions with `X_square`)

Interpretation: repeated convolution with `X_square` smooths the hard polygon indicator into a compactly supported algebraic field with controlled regularity.

---

## 3) Explicit algebraic construction (paper Sections 3–4)

The paper derives a closed form by introducing a bivariate building block `S_{2,a,b,delta}^{(n)}(x,y)`, expressed as finite alternating sums of shifted elementary polynomial primitives `A_{a,b}^{(n)}` (Eq. (7)–(8) in the paper).

High-level form:

`S_{2,a,b,delta}^{(n)}(x,y) = (1 / (4*delta^2)^n) * sum_{i=0..n} sum_{j=0..n} [(-1)^(i+j) * c_i * c_j * F_{i,j}(x,y)]`

where `F_{i,j}` are shifted evaluations of `A_{a,b}^{(n)}` and `c_i` are combinatorial coefficients induced by repeated box convolution.

### 3.1) Implicit geometric operators

The construction is organized through three objects:

1. **Implicit vertex**: translated copy of the central building block at polygon vertex `V`.
2. **Implicit edge**: oriented difference of endpoint implicit vertices.
3. **Implicit polygon**: signed sum of implicit edges around polygon boundary.

The key theorem (Theorem 3.4) shows this edge-sum representation equals the convolution-defined `B_{A,delta}^{(n)}`.

---

## 4) Derivation roadmap (implementation-oriented)

1. **Geometric integral view of convolution**  
   Evaluate `B_{A,delta}^{(n)}(P)` as overlap measure between `A` and translated kernel supports around point `P`.

2. **Finite-difference pattern from repeated box convolution**  
   Iterating square convolution yields alternating shifted polynomial sums; for `n=1,2,3`, masks appear as `2x2`, `3x3`, `4x4` patterns.

3. **Boundary decomposition**  
   Convert area expression into oriented boundary contributions: edge terms emerge as endpoint differences.

4. **Special/degenerate cases**  
   Axis-aligned and vertical/horizontal edge cases are handled explicitly (including formulations via smooth steps `H_n`, paper Eq. (12)) to maintain correctness across polygon configurations.

---

## 5) Fundamental properties

From convolution and explicit formulas:

- **Boundedness / non-negativity**: `0 <= B_{A,delta}^{(n)}(x,y) <= 1`.
- **Partition of unity**: if `{A_k}` partitions `R^2` up to measure-zero overlaps, then `sum_k B_{A_k,delta}^{(n)}(x,y) = 1`.
- **Local support**: finite polygons induce finite support neighborhoods.
- **Additivity on disjoint sets**: `B_{A union C,delta}^{(n)} = B_{A,delta}^{(n)} + B_{C,delta}^{(n)}` for `A intersect C = empty`.
- **Smoothness**: `B_{A,delta}^{(n)} in C^(n-1)`.

---

## 6) Parameter effects and practical guidance

### 6.1) Smoothness order `n`
- Larger `n`: smoother fields, broader effective transition, higher polynomial degree/cost.
- Smaller `n`: sharper transition, lower cost, less differentiability.

### 6.2) Smoothing scale `delta`
- Smaller `delta`: contours track polygon boundaries more tightly.
- Larger `delta`: stronger smoothing and wider blending zones.

Rule of thumb: choose `delta` as a fraction of the smallest local feature size; reduce grid spacing/sampling step as `delta` decreases.

### 6.3) Contour isovalue
For normalized occupancy-like fields, `tau = 0.5` is common, but application-specific thresholds may be preferable after blending/weighting.

### 6.4) Vertex orientation
Use consistent polygon orientation (typically CCW for outer boundaries). If holes are represented, orient inner boundaries oppositely to preserve sign conventions.

---

## 7) Code mapping in this repository

Primary implementation is in:

- `src/implicit_spline.py`

Expected responsibilities:

- evaluate elementary primitives `A_{a,b}^{(n)}`,
- assemble `S_{2,a,b,delta}^{(n)}` via finite shifted masks,
- construct implicit vertex/edge/polygon contributions,
- compute weighted sums and extract contours (including forms related to Eq. (5), Eq. (20) in the paper).

Suggested public parameters:

- `smoothness_order` (`n`),
- `delta`,
- polygon vertex order / winding policy,
- contour `isovalue`,
- sampling resolution / grid step.

---

## 8) Numerical and robustness notes

- **Stability near degeneracies**: guard near-zero edge lengths and nearly collinear triples.
- **Normalization checks**: verify partition-of-unity numerically on test partitions.
- **Regression tests**:
  - single rectangle (compare against separable 1D reference where applicable),
  - adjacent polygons (check continuity and sum-to-one),
  - varying `n,delta` sweep (check monotonic smoothness/blur behavior).
- **Performance**: precompute mask coefficients and reuse shifted evaluations when scanning many query points.

---

## 9) Minimal validation checklist

For any new implementation change:

1. `0 <= B <= 1` on sampled domain.
2. `sum_k B_k ~= 1` on partition test set.
3. No visible seams across shared edges for expected smoothness.
4. Sharper boundary behavior as `delta -> 0+` (with correspondingly refined sampling).
5. Continuity order consistent with `n` in numerical derivative probes.

---

## Reference

Li, Q. and Tian, J. (2009).  
**2D Piecewise Algebraic Splines for Implicit Modeling.**  
*ACM Transactions on Graphics*, 28(2), Article 13.  
DOI: https://doi.org/10.1145/1516522.1516524
