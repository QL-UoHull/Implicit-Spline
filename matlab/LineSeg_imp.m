function f = LineSeg_imp(x, y, x1, y1, x2, y2, delta, n)
%LineSeg_imp  Smooth implicit function for a finite directed line segment.
%
%   f = LineSeg_imp(x, y, x1, y1, x2, y2, delta, n)
%
%   Returns a smooth function that:
%     • is positive on the LEFT of the segment body (interior side),
%     • transitions smoothly to zero within the band of width delta
%       around the segment line,
%     • tapers smoothly to zero beyond both endpoints via a tangential gate.
%
%   Parameters
%   ----------
%   x, y         : query point(s) — scalar or any array
%   x1,y1,x2,y2  : directed segment endpoints (scalars)
%   delta        : transition/tapering width (default: 0.1)
%   n            : smoothness order for H (default: 2)
%
%   Algorithm
%   ---------
%   1. Compute the perpendicular signed distance Lperp = Lxy(…).
%   2. Compute the along-segment coordinate t_proj ∈ [0, len].
%   3. Build a tangential gate: gate = H(t_proj,delta,n)·H(len-t_proj,delta,n)
%      which is 1 in the segment interior and falls off smoothly at the ends.
%   4. Return f = Lperp * gate.
%
%   Degenerate segments (zero length) fall back to −Point_imp.
%
%   See also: Lxy, H, Point_imp, ImpSpline2D.

if nargin < 7 || isempty(delta), delta = 0.1; end
if nargin < 8 || isempty(n),     n     = 2;   end

dx  = x2 - x1;
dy  = y2 - y1;
len = sqrt(dx^2 + dy^2);

if len < eps
    % Degenerate: treat as a point, return negative proximity
    f = -Point_imp(x, y, x1, y1);
    return;
end

% Perpendicular signed distance to the supporting line (positive = left)
Lperp = (dx .* (y - y1) - dy .* (x - x1)) / len;

% Projection of query point onto the segment axis, in [0, len]
t_proj = (dx .* (x - x1) + dy .* (y - y1)) / len;

% Smooth tangential gate: 1 well inside the segment, 0 beyond endpoints
gate = H(t_proj, delta, n) .* H(len - t_proj, delta, n);

% Combined function: carries the perpendicular sign, vanishes at ends
f = Lperp .* gate;
end
