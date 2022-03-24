function q = plotVectors(o, prt, vec, col, scl)
%DRAWPARTICLES Draw arrows on the current figure

if ~exist('scale', 'var')
    scl = 'on';
end
if ~exist('col', 'var')
    col = 'b';
end

% Get the coordinates and arrows
if o.par_dim >= 1; rx = o.prt_pos(prt,1); ux = vec(:,1); end
if o.par_dim >= 2; ry = o.prt_pos(prt,2); uy = vec(:,2); end
if o.par_dim >= 3; rz = o.prt_pos(prt,3); uz = vec(:,3); end

hold('on');
switch o.par_dim
    case 1
        % Slant the arrows a bit for visibility
        uy = .1*ux;
        q = quiver( rx, 0.*rx, ux, uy, 'AutoScale', scl, 'Color', col );
    case 2
        q = quiver( rx, ry, ux, uy, 'AutoScale', scl, 'Color', col );
    case 3
        q = quiver( rx, ry, rz, ux, uy, uz, 'AutoScale', scl, 'Color', col );
    otherwise
        error('Must be in [1,3] dimensions!');
end
hold('off')

% Scale image
axis('image');
if o.par_dim >= 1; xlim([o.geo_min(1), o.geo_max(1)]); end
if o.par_dim >= 2; ylim([o.geo_min(2), o.geo_max(2)]); end
if o.par_dim >= 3; zlim([o.geo_min(3), o.geo_max(3)]); end

% Set figure background color
set(gca,'Color',[0,0,0]);

end