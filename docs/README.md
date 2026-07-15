# Theory and Derivations for 2D Piecewise Algebraic Splines

This document summarizes the mathematical theory and derivations behind **2D piecewise algebraic splines for implicit modeling** (Li & Tian, ACM TOG 2009). It explains the construction of analytically expressed bivariate spline basis functions built from arbitrary polygonal partitions, highlights the central formulas, and sketches the derivation steps so implementers and researchers can reproduce the results and link them to the code in this repository.

## Key concepts and goals

- **Objective**  
  Given a partition of \mathbb{R}^2 by polygons \(\{A_k\}\), construct basis functions \(B_{A_k,\delta}^{(n)}(x,y)\) that are:
  - piecewise polynomial and explicitly representable;
  - non‑negative and form a partition of unity;
  - locally supported for finite polygons;
  - additive across nonoverlapping polygons;
  - \(C^{n-1}\) continuous for any prescribed integer order \(n\ge1\).

- **Polygon smoothing parameter \(\delta\)**  
  Controls the blending width between inside/outside of polygons; smaller \(\delta\) yields contours closer to polygon boundaries.

## Core construction

1. **Polygon characteristic function**  
   For polygon \(A\), define
   

\[
   X_A(x,y)=\begin{cases}1&(x,y)\in A\

\[4pt]0&\text{otherwise.}\end{cases}
   \]



2. **Square kernel and iterative convolution**  
   Let \(X_\square\) be the characteristic function of a square of side \(2\delta\) centered at the origin. Define the order‑\(n\) basis by iterated convolution:
   

\[
   B_{A,\delta}^{(n)}(x,y)=\underbrace{X_A * X_\square * \cdots * X_\square}_{\text{\(n\) convolutions}}(x,y).
   \]


   This yields a piecewise polynomial function with \(C^{n-1}\) continuity.

3. **Analytic closed form via shifted building blocks**  
   The construction uses an explicit bivariate building block \(S_{2,a,b,\delta}^{(n)}(x,y)\) expressed as a finite linear combination of shifted elementary polynomials \(A_{a,b}^{(n)}\). Each polygonal basis is assembled from these blocks by summing signed implicit edges (differences of implicit vertices) around the polygon.

4. **Implicit vertex, implicit edge, implicit polygon**  
   - **Implicit vertex**: translated building block centered at a polygon vertex.  
   - **Implicit edge**: oriented difference of two implicit vertices; contributes the edge’s effect.  
   - **Implicit polygon**: signed sum of implicit edges around the polygon; equals the convolution result \(B_{A,\delta}^{(n)}\).

## Important formulas and building blocks

- **Elementary polynomial** \(A_{a,b}^{(n)}(x,y)\) — base polynomial pieces used to form \(S_{2,a,b,\delta}^{(n)}\). See the paper for explicit expressions and case expansions for small \(n\).
- **Composite block** \(S_{2,a,b,\delta}^{(n)}(x,y)\) — finite linear combination of shifted \(A_{a,b}^{(n)}\) evaluations with combinatorial coefficients; this block is the core analytic kernel.
- **Smooth unit step functions** \(H_n(t)\) — used to express axis‑aligned and rectangular cases and to relate to tensor‑product univariate splines.

## Derivation sketch

1. Interpret the iterated convolution geometrically as integrating the polygon characteristic over translated square kernels; repeated integration yields finite difference masks of shifted base polynomials.
2. Use induction on convolution order to derive the finite mask and coefficients (illustrated in the paper for \(n=1,2,3\)).
3. Reduce the polygon integral to oriented edge contributions by translating the base solution to polygon vertices and summing signed edge terms (Theorem 3.4).
4. Treat axis‑aligned and degenerate cases using \(H_n\) functions and explicit handling of orientations.

## Proven properties

- **Nonnegativity:** \(0\le B_{A,\delta}^{(n)}(x,y)\le 1\).  
- **Partition of unity:** If \(\{A_k\}\) partition \(\mathbb{R}^2\), then \(\sum_k B_{A_k,\delta}^{(n)}(x,y
