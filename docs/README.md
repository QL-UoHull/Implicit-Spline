# README – Theory and Derivations

This document summarizes the mathematical framework behind **2D Piecewise Algebraic Splines for Implicit Modeling**, introduced by:

> Li, Q. and Tian, J. (2009).  
> *2D Piecewise Algebraic Splines for Implicit Modeling*.  
> ACM Transactions on Graphics, 28(2), Article 13.  
> DOI: https://doi.org/10.1145/1516522.1516524. 

This repository implements the spline framework proposed in the paper and provides practical tools for implicit shape modeling using polygon-based spline functions.

---

# 1. Motivation

Classical tensor-product B-splines provide an elegant framework for parametric curve and surface design on regular grids. However, extending these ideas to arbitrary polygonal partitions remains challenging.

The goal of the implicit spline framework is to construct a set of smooth basis functions over an arbitrary polygonal partition $\{\Omega_k\}$ of $\mathbb{R}^2$

such that each basis function:

- is piecewise polynomial;
- admits an explicit analytical representation;
- is non-negative;
- has local support;
- possesses arbitrary smoothness;
- is additive over disjoint regions;
- forms a partition of unity.

More precisely, for every point

$$
\mathbf{p}\in\mathbb{R}^2
$$

the basis functions satisfy

$$
\sum_k B_k(\mathbf{p}) = 1.
$$


Unlike traditional box splines and simplex splines, the proposed basis functions can be evaluated directly using closed-form piecewise-polynomial expressions, making them attractive for practical implicit modeling applications.

---

# 2. Polygon-Based Spline Basis Functions

Given a polygon

$$
\Omega \subset \mathbb{R}^2,
$$

the objective is to construct a smooth function

$$
B_{\Omega,\delta}^{(n)}(x,y)
$$

that behaves like a softened indicator function of the polygon.

The parameter

$$
\delta > 0
$$

controls the width of the smoothing region, while

$$
n
$$

controls the smoothness order.

The resulting basis functions possess many of the same geometric properties as classical B-spline bases, while being defined directly on arbitrary polygonal domains.

---

# 3. Recursive Convolution Construction

Let

$$
\chi_{\Omega}(x,y)
$$

denote the characteristic function of polygon $\Omega$:

$$
\chi_{\Omega}(x,y)=
\begin{cases}
1,& (x,y)\in\Omega\\
0,& (x,y)\notin\Omega
\end{cases}
$$

Let

$$
\chi
$$

denote the characteristic function of a square centered at the origin with side length

$$
2\delta.
$$

The spline basis function is defined recursively as

$$
B^{(0)}_{\Omega,\delta}(x,y) = \chi_{\Omega}(x,y),
$$

and

$$
B^{(n)}_{\Omega,\delta}(x,y) = 
\frac{1}{4\delta^2}
\int_{\mathbb R^2}
B^{(n-1)}_{\Omega,\delta}(s,t)
\chi(s-x,t-y)
\,ds\,dt.
$$

Repeated convolution progressively smooths the polygon indicator function while preserving locality and positivity. Each convolution increases the smoothness by one order. 

---

# 4. Geometric Interpretation of the Smoothing Parameter

The parameter

$$
\delta
$$

controls how closely the resulting spline follows the original polygon.

### Small δ

- tight approximation to polygon boundaries;
- narrow blending regions;
- sharper geometric features.

### Large δ

- broader blending regions;
- smoother transitions;
- rounder implicit shapes.

The contour

$$
B_{\Omega,\delta}^{(n)}(x,y)=0.5
$$

typically provides a smooth geometric approximation of the original polygon boundary. 

---

# 5. Explicit Algebraic Representation

A major contribution of the paper is the derivation of explicit closed-form expressions for the spline basis functions.

The recursive convolution is transformed into a finite summation involving a family of auxiliary piecewise-polynomial functions

$$
A_{\alpha,\beta}^{(n)}(x,y).
$$

From these functions, the fundamental building block

$$
\Phi_{\alpha,\beta,\delta}^{(n)}(x,y)
$$

is constructed:

$$
\Phi_{\alpha,\beta,\delta}^{(n)}=
\frac{1}{(4\delta^2)^n}
\sum_{i=0}^{n}
\sum_{j=0}^{n}
(-1)^{i+j}
\binom{n}{i}
\binom{n}{j}
F_{ij}(x,y).
$$

The functions

$$
F_{ij}
$$

are shifted evaluations of the elementary polynomial primitives.

This formulation converts repeated convolution into a finite difference scheme that can be evaluated efficiently and accurately. 

---

# 6. Implicit Geometric Operators

The implementation is built on three fundamental geometric primitives.

## 6.1 Implicit Vertex

An implicit vertex represents a smoothed corner located at a polygon vertex.

Each implicit vertex is constructed from a translated copy of

$$
\Phi_{\alpha,\beta,\delta}^{(n)}.
$$

The vertex orientation is determined by the associated edge direction.

---

## 6.2 Implicit Edge

An implicit edge is formed from the difference of two implicit vertices.

Conceptually, an implicit edge behaves as a smooth indicator function of a half-space bounded by the corresponding polygon edge.

---

## 6.3 Implicit Polygon

An implicit polygon is obtained by summing signed implicit edge contributions around the polygon boundary.

For a polygon specified in counter-clockwise order,

$$
Q^{(n)}_\delta
$$

is defined as the signed sum of all implicit edges.

This formulation converts the original area-based construction into a computationally efficient boundary-based representation. 

---

# 7. Fundamental Theorem

A key result of the paper (Theorem 3.4) establishes that the explicit implicit-polygon representation is exactly equivalent to the recursive convolution definition.

Specifically,

$$
B^{(n)}_{\Omega,\delta}=
Q^{(n)}_\delta(\Omega).
$$

This theorem provides the mathematical foundation for efficient implementation because evaluating the spline requires only polygon boundary information rather than repeated numerical convolutions. 
---

# 8. Mathematical Properties

The resulting spline basis functions possess several important properties.

## Non-Negativity

$$
0
\le
B^{(n)}_{\Omega,\delta}(x,y)
\le
1.
$$

---

## Smoothness

$$
B^{(n)}_{\Omega,\delta}
\in
C^{n-1}.
$$

Increasing the order \(n\) produces increasingly smooth basis functions.

---

## Piecewise Polynomial Structure

Every spline basis function consists of a finite collection of polynomial patches.

This gives exact analytical evaluation and efficient implementation.

---

## Local Support

If the polygon is finite, the resulting spline basis function also has finite support.

---

## Additivity

For disjoint polygons,

$$
B^{(n)}_{\Omega_1\cup\Omega_2}=
B^{(n)}_{\Omega_1}
+
B^{(n)}_{\Omega_2}.
$$

---

## Partition of Unity

For a polygonal partition

$$
\{\Omega_k\},
$$

the spline basis functions satisfy

$$
\sum_k
B^{(n)}_{\Omega_k,\delta}(x,y)=
1.
$$

This property is fundamental for interpolation and blending operations. 
---

# 9. Relationship to Classical B-Splines

One particularly interesting result is that the proposed spline framework reduces to a tensor-product construction when the underlying polygon is a rectangle.

For the rectangle

$$
[x_0,x_1]\times[y_0,y_1],
$$

the bivariate spline can be expressed as

$$
B^{(n)}= B^{(n)}_{[x_0,x_1]}*B^{(n)}_{[y_0,y_1]}.
$$

Thus the proposed polygonal spline can be viewed as a geometric generalization of tensor-product B-splines from rectangular domains to arbitrary polygonal regions. 

---

# 10. Implicit Curve Design

The contour

$$
B^{(n)}_{\Omega,\delta}(x,y)=0.5
$$

defines a smooth implicit curve associated with the control polygon.

Designers create shapes simply by specifying polygon vertices.

Important observations include:

- smaller values of \(\delta\) produce contours closer to the control polygon;
- larger values of \(\delta\) generate smoother shapes;
- holes can be represented using suitably oriented polygon loops;
- complex branching structures can be modeled using a single control polygon;
- the resulting curve generally exhibits variation-diminishing behavior when \(\delta\) is sufficiently small. 

---

# 11. Constructive Implicit Shape Modeling

A major advantage of the spline basis is that arbitrary implicit functions may be blended together.

Given local implicit functions

$$
F_k(x,y),
$$

defined on polygons

$$
\Omega_k,
$$

a global implicit function may be constructed as

$$
F(x,y)=
\sum_k
F_k(x,y)
B_k(x,y).
$$

This representation provides:

- local control;
- smooth transitions;
- partition-of-unity blending;
- shape-preserving composition.

The formulation serves as an implicit analogue of classical spline-based freeform design. 

---

# 12. Implicit Surface Design

The paper demonstrates several 3D modeling techniques constructed from 2D implicit spline curves.

## Implicit Spline Surfaces

Cross-sectional implicit curves are blended along a spatial direction using spline basis functions.

$$
F(x,y,z)=
\sum_k
C_k(x,y)
B_k(z).
$$

---

## Extruded Implicit Surfaces

Implicit profile curves may be extruded along arbitrary spatial paths by mapping

$$
\mathbb{R}^3 \rightarrow \mathbb{R}^2.
$$

---

## Surfaces of Revolution

Rotating an implicit spline profile around an axis directly generates smooth implicit surfaces of revolution.

---

## Shape Reconstruction

The framework can also be used to reconstruct smooth implicit surfaces from sampled geometric data. 

---

# 13. Implementation Mapping

| Mathematical Concept | Repository Implementation |
|----------------------|---------------------------|
| \(A_{\alpha,\beta}^{(n)}\) | Elementary polynomial primitive |
| \(\Phi_{\alpha,\beta,\delta}^{(n)}\) | Core spline building block |
| Implicit vertex | Smoothed corner operator |
| Implicit edge | Oriented boundary contribution |
| Implicit polygon | Polygon spline basis function |
| \(B^{(n)}_{\Omega,\delta}=0.5\) | Implicit contour extraction |
| \(\sum F_k B_k\) | Global implicit model assembly |

---

# 14. Practical Guidelines

## Choosing Smoothness Order

- `n = 1` → C⁰ continuity
- `n = 2` → C¹ continuity
- `n = 3` → C² continuity (recommended)
- larger values provide smoother transitions but increase polynomial degree

---

## Choosing δ

A useful rule of thumb is

```text
δ ≈ 10% – 25% of the smallest geometric feature size
