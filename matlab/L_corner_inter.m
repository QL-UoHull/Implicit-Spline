function f = L_corner_inter(x, y, xA, yA, xB, yB, xC, yC, delta, n)
%L_corner_inter  Smooth blending function for a general polygon vertex.
%
%   f = L_corner_inter(x, y, xA, yA, xB, yB, xC, yC, delta, n)
%
%   Builds the corner basis function at vertex B, where two polygon edges
%   meet: incoming edge A→B and outgoing edge B→C.
%
%   The function equals:
%     f = H(L_AB, delta, n) * H(L_BC, delta, n)
%
%   where L_AB and L_BC are the signed distances to the two edge lines.
%   This product is:
%     • 1   deep inside the polygon (both distances >> delta),
%     • 0   near either edge (one distance ≈ 0),
%     • C^n smooth everywhere.
%
%   For specialised corner shapes see Square_Angle_inter (right angles)
%   and U_Angle_inter (obtuse / reflex angles).
%
%   Parameters
%   ----------
%   x, y                : query point(s)
%   xA,yA               : vertex A (tail of incoming edge)
%   xB,yB               : vertex B (the corner)
%   xC,yC               : vertex C (head of outgoing edge)
%   delta               : transition width (default: 0.1)
%   n                   : smoothness order (default: 2)
%
%   See also: Square_Angle_inter, U_Angle_inter, ImpSpline2D, H, Lxy.

if nargin < 9  || isempty(delta), delta = 0.1; end
if nargin < 10 || isempty(n),     n     = 2;   end

L1 = Lxy(x, y, xA, yA, xB, yB);   % signed dist to line through A→B
L2 = Lxy(x, y, xB, yB, xC, yC);   % signed dist to line through B→C

h1 = H(L1, delta, n);
h2 = H(L2, delta, n);

% Product ensures the function vanishes when near either boundary edge
f = h1 .* h2;
end
