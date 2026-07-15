function L = Lxy00(x, y, x1, y1, x2, y2)
%Lxy00  Unnormalised signed linear function for directed line (x1,y1)→(x2,y2).
%
%   L = Lxy00(x, y, x1, y1, x2, y2)
%
%   Returns the same signed quantity as Lxy but WITHOUT dividing by the
%   edge length.  Useful when only the sign or zero-set matters, or when
%   the expression is kept as a polynomial factor in algebraic
%   computations.
%
%   Formula:
%       dx = x2 - x1;  dy = y2 - y1
%       L  = dx*(y - y1) - dy*(x - x1)
%
%   Sign convention: positive to the LEFT of the directed edge (same as
%   Lxy), so interior points of a CCW polygon give L > 0 for every edge.
%
%   All inputs may be scalars or numeric arrays of the same size.
%
%   See also: Lxy, ImpSpline2D.

dx = x2 - x1;
dy = y2 - y1;
L  = dx .* (y - y1) - dy .* (x - x1);
end
