%Demo_v1  Original-style demo (legacy; see Demo.m for the full version).
%
%   Demonstrates the implicit spline for a single hard-coded polygon using
%   the simplified DrawImpSpline_V1 visualisation.
%
%   Retained for reference.  Use Demo.m for multi-example demonstrations
%   with improved visualisation.
%
%   See also: Demo, ImpSpline2D, DrawImpSpline_V1.

clear; clc; close all;

% Parameters
delta = 0.12;
n     = 2;
N     = 150;

% Unit square polygon
P = [0 0; 1 0; 1 1; 0 1];

fprintf('Demo_v1: drawing implicit spline for a unit square.\n');
DrawImpSpline_V1(P, delta, n, N);
set(gcf, 'Name', 'Demo_v1');
fprintf('Done.\n');
