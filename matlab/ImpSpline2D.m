function f = ImpSpline2D(x, y, P, delta, n)
%ImpSpline2D  2D piecewise algebraic implicit spline for a polygon.
%
%   f = ImpSpline2D(x, y, P, delta, n)
%
%   Evaluates the smooth implicit function at query point(s) (x, y) for the
%   polygon whose vertices are given in P.
%
%   Returned values satisfy:
%     f ≈ 1   deep inside the polygon (all edge distances ≥ delta)
%     f = 0   on the polygon boundary (at least one edge distance = 0)
%     f ≈ 0   outside the polygon (at least one edge distance ≤ 0)
%
%   The function is the product of per-edge smooth step functions:
%
%       f(x,y) = ∏_{i=1}^{m}  H( L_i(x,y), delta, n )
%
%   where L_i is the normalised signed distance from (x,y) to edge i
%   (positive on the interior side).  This makes f a piecewise
%   polynomial of total degree m*(2n+1) that is C^n near each edge.
%
%   Parameters
%   ----------
%   x, y  : query point(s) — scalars or arrays of matching shape
%   P     : m×2 array of polygon vertices.  Vertices may be given in
%           either CW or CCW order; the function auto-corrects to CCW.
%           The polygon need not be explicitly closed (first ≠ last row).
%   delta : transition bandwidth.  Larger values give wider smooth zones.
%           Recommended: 5–30% of the polygon's inscribed-circle radius.
%           (default: 0.1)
%   n     : smoothness order.  The result is C^n continuous near each
%           polygon edge. (default: 2)
%
%   Returns
%   -------
%   f : same shape as x/y, values in [0, 1]
%
%   Notes
%   -----
%   • The construction is exact for convex polygons.  For non-convex
%     polygons some interior regions near reflex vertices may receive
%     reduced values; increase delta or pre-process the polygon.
%   • For interactive input of polygon vertices see dataFrGr.
%   • For batch grid evaluation and plotting see DrawImpSpline.
%
%   Examples
%   --------
%   % Evaluate on a grid for a unit square
%   P = [0 0; 1 0; 1 1; 0 1];
%   [X, Y] = meshgrid(linspace(-0.2, 1.2, 200));
%   F = ImpSpline2D(X, Y, P, 0.1, 2);
%   contourf(X, Y, F, 20); colorbar; axis equal;
%
%   Reference
%   ---------
%   Li, Q. & Tian, J. (2009). 2D Piecewise Algebraic Splines for Implicit
%   Modeling. ACM Transactions on Graphics, 28(3).
%   DOI: 10.1145/1516522.1516524
%
%   See also: H, Lxy, DrawImpSpline, dataFrGr.

if nargin < 4 || isempty(delta), delta = 0.1; end
if nargin < 5 || isempty(n),     n     = 2;   end

% ── Input validation ──────────────────────────────────────────────────────
P = double(P);
if size(P, 2) ~= 2
    error('ImpSpline2D:badP', ...
        'P must be an m×2 array of (x,y) vertices (got size [%d %d]).', ...
        size(P, 1), size(P, 2));
end
m = size(P, 1);
if m < 3
    error('ImpSpline2D:tooFewVertices', ...
        'Polygon must have at least 3 vertices (got %d).', m);
end
if delta <= 0
    error('ImpSpline2D:badDelta', 'delta must be positive (got %g).', delta);
end

% ── Ensure counter-clockwise orientation (positive interior → L_i > 0) ──
% Shoelace signed area: positive ↔ CCW
xi = P(:, 1);  yi = P(:, 2);
xj = circshift(xi, -1);  yj = circshift(yi, -1);
signed_area2 = sum(xi .* yj - xj .* yi);   % twice the signed area
if signed_area2 < 0
    P = P(end:-1:1, :);                     % reverse to CCW
end

% ── Assemble product of per-edge smooth step functions ────────────────────
% BUG FIX: earlier versions used an undeclared variable 'zz' at this point
% instead of the function output 'f'.  The accumulator is now consistently
% named 'f' throughout.
f = ones(size(x));

for i = 1 : m
    j = mod(i, m) + 1;                      % next vertex index (wraps around)
    L_i = Lxy(x, y, P(i,1), P(i,2), P(j,1), P(j,2));
    f   = f .* H(L_i, delta, n);
end
end
