function [Z, X, Y] = DrawImpSpline(P, delta, n, N, bbox, do_plot)
%DrawImpSpline  Evaluate and optionally visualise the implicit spline.
%
%   [Z, X, Y] = DrawImpSpline(P, delta, n, N, bbox, do_plot)
%
%   Evaluates ImpSpline2D on a regular grid and produces a filled-contour
%   plot with an iso-level overlay at f = 0.5.
%
%   This function SEPARATES computation from visualisation: it always
%   returns the grid arrays (Z, X, Y) and only draws figures when
%   do_plot = true.  The core ImpSpline2D function is not called with any
%   plotting side-effects.
%
%   Parameters
%   ----------
%   P       : m×2 polygon vertex array
%   delta   : transition bandwidth (default: 0.1)
%   n       : smoothness order (default: 2)
%   N       : grid resolution (N×N points, default: 200)
%   bbox    : [xmin xmax ymin ymax] evaluation window.
%             Pass [] to auto-compute with a 20% margin around P.
%   do_plot : logical flag — true to display figures (default: true)
%
%   Returns
%   -------
%   Z : N×N array of implicit function values
%   X : N×N x-coordinates of the evaluation grid
%   Y : N×N y-coordinates of the evaluation grid
%
%   Examples
%   --------
%   P = [0 0; 2 0; 1 1.8];
%   DrawImpSpline(P, 0.15, 2, 200, [], true);
%
%   % Retrieve values without plotting
%   [Z, X, Y] = DrawImpSpline(P, 0.1, 2, 100, [], false);
%   surf(X, Y, Z);
%
%   See also: ImpSpline2D, DrawImpSpline_V1, dataFrGr.

if nargin < 2 || isempty(delta),   delta   = 0.1;  end
if nargin < 3 || isempty(n),       n       = 2;    end
if nargin < 4 || isempty(N),       N       = 200;  end
if nargin < 5,                     bbox    = [];   end
if nargin < 6 || isempty(do_plot), do_plot = true; end

% ── Auto-compute bounding box if not supplied ────────────────────────────
if isempty(bbox)
    pad  = 0.2 * max(max(P(:,1)) - min(P(:,1)), max(P(:,2)) - min(P(:,2)));
    pad  = max(pad, delta);          % ensure at least one transition width
    bbox = [min(P(:,1)) - pad, max(P(:,1)) + pad, ...
            min(P(:,2)) - pad, max(P(:,2)) + pad];
end

% ── Build evaluation grid ─────────────────────────────────────────────────
x_vec = linspace(bbox(1), bbox(2), N);
y_vec = linspace(bbox(3), bbox(4), N);
[X, Y] = meshgrid(x_vec, y_vec);

% ── Evaluate implicit spline (vectorised, no plotting side-effects) ────────
Z = ImpSpline2D(X, Y, P, delta, n);

% ── Optional visualisation ────────────────────────────────────────────────
if do_plot
    figure('Name', 'Implicit Spline', 'NumberTitle', 'off');

    % Subplot 1: filled contour + iso-level + polygon outline
    subplot(1, 2, 1);
    contourf(X, Y, Z, 20, 'LineColor', 'none');
    colorbar;
    hold on;
    contour(X, Y, Z, [0.5 0.5], 'w', 'LineWidth', 2);
    P_closed = [P; P(1,:)];
    plot(P_closed(:,1), P_closed(:,2), 'r--', 'LineWidth', 1.5);
    plot(P(:,1), P(:,2), 'r.', 'MarkerSize', 10);
    axis equal; axis tight;
    xlabel('x'); ylabel('y');
    title(sprintf('Contour  (\\delta = %.3g,  n = %d)', delta, n));

    % Subplot 2: surface plot
    subplot(1, 2, 2);
    surf(X, Y, Z, 'EdgeColor', 'none');
    colorbar;
    xlabel('x'); ylabel('y'); zlabel('f(x,y)');
    title('Surface');
    view(-30, 35);
end
end
