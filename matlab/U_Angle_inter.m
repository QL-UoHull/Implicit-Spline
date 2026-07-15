function f = U_Angle_inter(x, y, xA, yA, xB, yB, xC, yC, delta, n)
%U_Angle_inter  Smooth corner function for obtuse or near-straight vertices.
%
%   f = U_Angle_inter(x, y, xA, yA, xB, yB, xC, yC, delta, n)
%
%   Specialised version of L_corner_inter for corners where the interior
%   angle at B is greater than 90° (obtuse) or approaches 180° (nearly
%   flat).  At such vertices the product H1*H2 remains very small over a
%   wide region even inside the polygon; the inclusive-OR formula below
%   corrects this by giving higher values in the angular region between
%   the two edges.
%
%   Construction
%   ------------
%   Instead of h1*h2 (AND), use the probabilistic-OR combination:
%
%       f = h1 + h2 − h1 * h2
%
%   which is always ≥ max(h1,h2) and equals 1 when either hi = 1.
%   This ensures that deep inside the wide-angle region f ≈ 1 rather than
%   remaining depressed by the product.
%
%   Parameters
%   ----------
%   x, y               : query point(s)
%   xA,yA,xB,yB,xC,yC  : vertex coordinates (B is the corner vertex)
%   delta              : transition width (default: 0.1)
%   n                  : smoothness order (default: 2)
%
%   See also: L_corner_inter, Square_Angle_inter, ImpSpline2D.

if nargin < 9  || isempty(delta), delta = 0.1; end
if nargin < 10 || isempty(n),     n     = 2;   end

L1 = Lxy(x, y, xA, yA, xB, yB);
L2 = Lxy(x, y, xB, yB, xC, yC);

h1 = H(L1, delta, n);
h2 = H(L2, delta, n);

% Inclusive-OR: avoids the "pinched" interior that the AND product
% produces for wide angles.
f = h1 + h2 - h1 .* h2;
end
