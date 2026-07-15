%Demo  Demonstration of the 2D Piecewise Algebraic Implicit Spline.
%
%   Runs four examples that showcase the main features of the
%   Li & Tian (2009) implicit-spline construction:
%
%     Example 1 : Unit square (right-angle corners)
%     Example 2 : Equilateral triangle (acute corners)
%     Example 3 : Regular pentagon
%     Example 4 : Irregular pentagon — sweeping delta to show bandwidth effect
%
%   Run from the  matlab/  directory or add it to the MATLAB path first:
%       addpath('path/to/matlab');
%
%   All parameters are set at the top of each section so they are easy to
%   adjust for exploration.
%
%   See also: ImpSpline2D, DrawImpSpline, dataFrGr.

clear; clc; close all;

%% ── Shared parameters ────────────────────────────────────────────────────
delta = 0.12;    % transition bandwidth
n     = 2;       % smoothness order (C^n near each edge)
N     = 200;     % grid resolution (N×N points)

fprintf('=== 2D Implicit Spline Demo ===\n\n');

%% ── Example 1: Unit square ───────────────────────────────────────────────
fprintf('Example 1: unit square\n');
P_sq = [0 0; 1 0; 1 1; 0 1];

DrawImpSpline(P_sq, delta, n, N, [], true);
set(gcf, 'Name', 'Demo — Example 1: Square');

%% ── Example 2: Equilateral triangle ─────────────────────────────────────
fprintf('Example 2: equilateral triangle\n');
P_tri = [0, 0; 2, 0; 1, sqrt(3)];

DrawImpSpline(P_tri, delta, n, N, [], true);
set(gcf, 'Name', 'Demo — Example 2: Triangle');

%% ── Example 3: Regular pentagon ─────────────────────────────────────────
fprintf('Example 3: regular pentagon\n');
theta  = linspace(pi/2, pi/2 + 2*pi, 6);   % start at top, CCW
theta  = theta(1:5);
P_pent = [cos(theta)', sin(theta)'];

DrawImpSpline(P_pent, delta, n, N, [], true);
set(gcf, 'Name', 'Demo — Example 3: Pentagon');

%% ── Example 4: Effect of delta (bandwidth) ───────────────────────────────
fprintf('Example 4: sweeping delta  [0.05, 0.15, 0.30]\n');
P_irr  = [0 0; 2 0; 2.5 0.8; 1.5 1.8; -0.2 1.2];   % irregular pentagon
deltas = [0.05, 0.15, 0.30];

figure('Name', 'Demo — Example 4: delta sweep', ...
       'NumberTitle', 'off', 'Position', [100 100 1100 350]);

for k = 1:3
    [Z, X, Y] = DrawImpSpline(P_irr, deltas(k), n, N, [], false);
    subplot(1, 3, k);
    contourf(X, Y, Z, 20, 'LineColor', 'none'); colorbar; hold on;
    contour(X, Y, Z, [0.5 0.5], 'w', 'LineWidth', 2);
    P_closed = [P_irr; P_irr(1,:)];
    plot(P_closed(:,1), P_closed(:,2), 'r--', 'LineWidth', 1.5);
    axis equal; axis tight;
    xlabel('x'); ylabel('y');
    title(sprintf('\\delta = %.2f', deltas(k)));
end

fprintf('\nDemo complete.\n');
