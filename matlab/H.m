function h = H(t, delta, n)
%H  Smooth Heaviside-like step function of order n.
%
%   h = H(t, delta, n)
%
%   Returns a smooth monotone function that transitions from 0 (at t <= 0)
%   to 1 (at t >= delta), using a degree-(2*n+1) Bernstein polynomial that
%   satisfies H^(k)(0) = 0 and H^(k)(delta) = 0 for k = 1 ... n.
%   The result is therefore C^n continuous at both transition points.
%
%   This function is fully vectorised and accepts scalars or arrays.
%
%   Parameters
%   ----------
%   t     : input value(s) — scalar or any numeric array
%   delta : transition width; must be positive (default: 1)
%   n     : smoothness order; result is C^n at both ends (default: 2)
%           n=1 → classic cubic smoothstep (degree 3)
%           n=2 → quintic smoothstep (degree 5), recommended default
%           n=3 → degree-7 smoothstep
%
%   Returns
%   -------
%   h : same size as t, values in [0, 1]
%
%   Examples
%   --------
%   % Plot the step for three smoothness orders
%   t = linspace(-0.2, 1.2, 300);
%   figure; hold on;
%   for k = 1:3
%       plot(t, H(t, 1, k), 'DisplayName', sprintf('n=%d', k));
%   end
%   legend; xlabel('t'); ylabel('H(t)'); title('Smooth step function');
%
%   Reference
%   ---------
%   Li, Q. & Tian, J. (2009). 2D Piecewise Algebraic Splines for Implicit
%   Modeling. ACM Transactions on Graphics, 28(3).

if nargin < 2 || isempty(delta), delta = 1; end
if nargin < 3 || isempty(n),     n     = 2; end

if delta <= 0
    error('H:invalidDelta', 'delta must be positive (received %g).', delta);
end
if ~(isscalar(n) && n >= 1 && floor(n) == n)
    error('H:invalidN', 'n must be a positive integer (received %g).', n);
end

% Map to normalised variable s in [0, 1]
s = min(max(t / delta, 0), 1);

% Bernstein-polynomial sum:  H_n(s) = sum_{k=n+1}^{2n+1} C(2n+1,k) s^k (1-s)^{2n+1-k}
% This formula evaluates correctly at s=0 (gives 0) and s=1 (gives 1).
h = zeros(size(s));
for k = (n + 1) : (2 * n + 1)
    h = h + nchoosek(2*n + 1, k) .* (s .^ k) .* ((1 - s) .^ (2*n + 1 - k));
end
end
