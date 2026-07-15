function DrawImpSpline_V1(P, delta, n, N)
%DrawImpSpline_V1  Simplified legacy visualisation (contourf only).
%
%   DrawImpSpline_V1(P, delta, n, N)
%
%   DEPRECATED — use DrawImpSpline instead, which separates computation
%   from visualisation, supports bounding-box control, returns grid data,
%   and adds a surface subplot.
%
%   This version is retained for reference / backward compatibility.
%   It evaluates the implicit spline on an N×N grid, then draws a
%   filled-contour plot with the 0.5 iso-level highlighted.
%
%   Parameters
%   ----------
%   P     : m×2 polygon vertices
%   delta : transition bandwidth (default: 0.1)
%   n     : smoothness order (default: 2)
%   N     : grid resolution (default: 100)
%
%   See also: DrawImpSpline, ImpSpline2D.

if nargin < 2 || isempty(delta), delta = 0.1; end
if nargin < 3 || isempty(n),     n     = 2;   end
if nargin < 4 || isempty(N),     N     = 100; end

% Auto bounding box
pad  = 0.2 * max(max(P(:,1)) - min(P(:,1)), max(P(:,2)) - min(P(:,2)));
pad  = max(pad, delta);
x_vec = linspace(min(P(:,1)) - pad, max(P(:,1)) + pad, N);
y_vec = linspace(min(P(:,2)) - pad, max(P(:,2)) + pad, N);
[X, Y] = meshgrid(x_vec, y_vec);

Z = ImpSpline2D(X, Y, P, delta, n);

figure;
contourf(X, Y, Z, 20, 'LineColor', 'none');
colorbar; hold on;
contour(X, Y, Z, [0.5 0.5], 'w', 'LineWidth', 2);
P_closed = [P; P(1,:)];
plot(P_closed(:,1), P_closed(:,2), 'r--', 'LineWidth', 1.5);
axis equal; axis tight;
xlabel('x'); ylabel('y');
title(sprintf('Implicit Spline (V1)  \\delta=%.3g  n=%d', delta, n));
end
