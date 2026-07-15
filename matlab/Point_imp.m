function d = Point_imp(x, y, px, py)
%Point_imp  Squared Euclidean distance from (x,y) to fixed point (px,py).
%
%   d = Point_imp(x, y, px, py)
%
%   Returns (x - px)^2 + (y - py)^2, a non-negative polynomial in x and y.
%   This is useful as a vertex-proximity primitive when building implicit
%   functions: near the vertex (px,py) the value approaches zero, allowing
%   smooth blending of adjacent edge functions.
%
%   All inputs may be scalars or numeric arrays of the same size.
%
%   See also: LineSeg_imp, L_corner_inter, ImpSpline2D.

d = (x - px).^2 + (y - py).^2;
end
