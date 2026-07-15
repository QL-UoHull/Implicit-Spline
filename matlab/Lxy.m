function L = Lxy(x, y, x1, y1, x2, y2)
%Lxy  Normalised signed distance from (x,y) to directed line (x1,y1)→(x2,y2).
%
%   L = Lxy(x, y, x1, y1, x2, y2)
%
%   Returns the perpendicular signed distance measured so that points to
%   the LEFT of the directed edge receive a POSITIVE value.  For a polygon
%   whose vertices are listed counter-clockwise (CCW), this means interior
%   points yield L > 0 for every edge.
%
%   Formula:
%       dx = x2 - x1;  dy = y2 - y1;  len = norm([dx, dy])
%       L  = (dx*(y - y1) - dy*(x - x1)) / len
%
%   All inputs may be scalars or numeric arrays of the same size
%   (element-wise evaluation).
%
%   Parameters
%   ----------
%   x, y         : query point(s)
%   x1,y1,x2,y2  : directed edge endpoints (scalars)
%
%   Returns
%   -------
%   L : same size as x/y
%
%   Notes
%   -----
%   • A degenerate edge (zero length) emits a warning and returns zeros.
%   • Use Lxy00 when the unnormalised value (no division by length) is
%     preferred, e.g. for polynomial manipulation.
%
%   See also: Lxy00, ImpSpline2D, H.

dx  = x2 - x1;
dy  = y2 - y1;
len = sqrt(dx^2 + dy^2);

if len < eps
    warning('Lxy:degenerateEdge', ...
        'Edge (%g,%g)->(%g,%g) has near-zero length; returning zeros.', ...
        x1, y1, x2, y2);
    L = zeros(size(x));
    return;
end

L = (dx .* (y - y1) - dy .* (x - x1)) / len;
end
