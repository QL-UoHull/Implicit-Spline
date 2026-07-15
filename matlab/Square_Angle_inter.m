function f = Square_Angle_inter(x, y, xA, yA, xB, yB, xC, yC, delta, n)
%Square_Angle_inter  Smooth corner function for a right-angle (90°) vertex.
%
%   f = Square_Angle_inter(x, y, xA, yA, xB, yB, xC, yC, delta, n)
%
%   Specialised version of L_corner_inter for corners where edges A→B and
%   B→C are (approximately) perpendicular.  The product formula is
%   augmented with a proximity weight centred on vertex B so that the
%   implicit iso-level rounds the corner more uniformly.
%
%   Construction
%   ------------
%   Let L1, L2 be the signed distances to the two edge lines and let
%   r_sq = (x-xB)^2 + (y-yB)^2 be the squared proximity to the corner.
%
%       w      = H(L1, delta, n) * H(L2, delta, n)   [product from L_corner]
%       corner = H(r_sq, (2*delta)^2, n)              [radial fade near B]
%       f      = w * (1 - corner) + corner            [blend: product far, 1 near B]
%
%   Near B (r_sq → 0) the radial fade dominates (f → 1), while far from B
%   the per-edge gates dominate.  This prevents the iso-level from being
%   pinched at right-angle corners.
%
%   Parameters
%   ----------
%   x, y             : query point(s)
%   xA,yA,xB,yB,xC,yC : vertex coordinates (B is the corner vertex)
%   delta            : transition width (default: 0.1)
%   n                : smoothness order (default: 2)
%
%   See also: L_corner_inter, U_Angle_inter, ImpSpline2D.

if nargin < 9  || isempty(delta), delta = 0.1; end
if nargin < 10 || isempty(n),     n     = 2;   end

L1 = Lxy(x, y, xA, yA, xB, yB);
L2 = Lxy(x, y, xB, yB, xC, yC);

h1 = H(L1, delta, n);
h2 = H(L2, delta, n);
w  = h1 .* h2;

% Squared distance to the corner vertex B
r_sq = Point_imp(x, y, xB, yB);

% Radial fade: transitions from 1 (very near B) to 0 (away from B)
% Threshold set at (2*delta)^2 so fade covers two transition widths
corner_fade = 1 - H(r_sq, (2 * delta)^2, n);

% Blend: near B favour 1 (full inside); away from B use edge product
f = w .* (1 - corner_fade) + corner_fade;
end
