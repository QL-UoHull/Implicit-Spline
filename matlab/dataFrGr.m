function P = dataFrGr(save_file)
%dataFrGr  Interactively capture polygon vertices from a figure window.
%
%   P = dataFrGr()
%   P = dataFrGr(save_file)
%
%   Opens a figure and lets the user click polygon vertices using the mouse.
%   Captured vertices are returned as an m×2 array.
%
%   Controls
%   --------
%   Left-click  : add a vertex at the clicked position
%   Right-click : close the polygon and return
%   Enter key   : close the polygon and return (alternative to right-click)
%
%   Parameters
%   ----------
%   save_file : (optional) file path to save the vertex list as plain text.
%               Each line contains one vertex: "x  y" with 6 decimal places.
%               If omitted or empty, no file is written.
%               Pass a full path to avoid ambiguity (e.g. '../data/pts.txt').
%
%   Returns
%   -------
%   P : m×2 double array.  Returns an empty 0×2 array if no vertices were
%       captured (e.g. figure closed immediately).
%
%   Notes
%   -----
%   • The original version wrote unconditionally to 'temp.dat' in the
%     current directory and re-read from it, making scripts brittle when
%     run from different working directories.  That hard-coded path has
%     been removed; callers may supply an explicit save_file if persistence
%     is needed.
%   • Mouse-button behaviour (which button closes the loop) is now
%     consistent: button == 3 (right-click) always closes, matching the
%     printed instructions.
%
%   Examples
%   --------
%   P = dataFrGr();                   % interactive, no file saved
%   P = dataFrGr('../data/pts.txt');  % interactive, vertices saved
%
%   See also: ImpSpline2D, DrawImpSpline.

if nargin < 1, save_file = ''; end

% ── Set up figure ─────────────────────────────────────────────────────────
hFig = figure('Name', 'Polygon Input — click vertices', ...
               'NumberTitle', 'off');
ax = axes('Parent', hFig);
axis(ax, [-5 5 -5 5]);
axis(ax, 'equal');
grid(ax, 'on');
hold(ax, 'on');
xlabel(ax, 'x'); ylabel(ax, 'y');
title(ax, 'Left-click: add vertex  |  Right-click or Enter: finish');
set(hFig, 'Pointer', 'crosshair');

P = zeros(0, 2);

fprintf('dataFrGr: click vertices in the figure.\n');
fprintf('  Left-click  → add vertex\n');
fprintf('  Right-click → finish\n');

while true
    % ginput(1) returns [] for key presses that map to 'enter' or
    % when the figure is closed; use try/catch for robustness.
    try
        [xi, yi, button] = ginput(1);
    catch
        break;   % figure was closed
    end

    if isempty(xi)
        % Enter / Return key pressed
        break;
    end

    if button == 3
        % Right-click: close polygon
        break;
    end

    if button ~= 1
        % Ignore any other mouse button (e.g. scroll wheel)
        continue;
    end

    % ── Left-click: add vertex ────────────────────────────────────────────
    P = [P; xi, yi]; %#ok<AGROW>

    plot(ax, xi, yi, 'ro', ...
         'MarkerSize', 7, 'MarkerFaceColor', [0.8 0 0], ...
         'MarkerEdgeColor', 'k');

    if size(P, 1) > 1
        plot(ax, P(end-1:end, 1), P(end-1:end, 2), ...
             'b-', 'LineWidth', 1.5);
    end
    drawnow;
end

% ── Close polygon outline if enough points ───────────────────────────────
if size(P, 1) >= 3
    plot(ax, [P(end,1) P(1,1)], [P(end,2) P(1,2)], ...
         'b-', 'LineWidth', 1.5);
    drawnow;
end

if size(P, 1) < 3
    warning('dataFrGr:tooFewPoints', ...
        'Only %d vertex/vertices captured; a valid polygon needs ≥ 3.', ...
        size(P, 1));
end

% ── Save to file if requested ─────────────────────────────────────────────
if ~isempty(save_file)
    fid = fopen(save_file, 'w');
    if fid < 0
        warning('dataFrGr:cannotWrite', ...
            'Cannot open "%s" for writing; vertex list not saved.', save_file);
    else
        fprintf(fid, '%.6f %.6f\n', P');
        fclose(fid);
        fprintf('dataFrGr: %d vertices saved to %s\n', size(P,1), save_file);
    end
end
end
