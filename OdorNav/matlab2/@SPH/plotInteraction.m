function plotInteraction(o, cl1, cl2, adj)
%PLOTINTERACTION Draw the interactions between fluid particles on plot

if ~exist('cl1', 'var')
    cl1 = 'all';
end
if ~exist('cl2', 'var')
    cl2 = 'all';
end
if ~exist('adj', 'var')
    adj = 'neighbour';
end

% Get the type of adjacency links
switch adj
    case {'neighbour', 'nbr'}
        [i, j] = find(o.hsh_prt_adj);
    case {'interacting', 'int'}
        i = o.cal_int_i;
        j = o.cal_int_j;
    otherwise
        error('Invalid adjacency type');
end

% Mask the links according to their types
m = true(length(i), 1);
switch cl1
    case 'all'
        m = m & o.prt_sim(i);
    case 'fluid'
        m = m & o.prt_fld(i);
    case {'boundary', 'border'}
        m = m & o.prt_bdr(i);
    case 'inlet'
        m = m & o.prt_inl(i);
    case 'outlet'
        m = m & o.prt_out(i);
    otherwise
        error('Invalid particle type');
end
switch cl2
    case 'all'
        m = m & o.prt_sim(j);
    case 'fluid'
        m = m & o.prt_fld(j);
    case {'boundary', 'border'}
        m = m & o.prt_bdr(j);
    case 'inlet'
        m = m & o.prt_inl(j);
    case 'outlet'
        m = m & o.prt_out(j);
    otherwise
        error('Invalid particle type');
end

i = i(m);
j = j(m);

hold('on');
switch o.par_dim
    case 1
        % Draw triangle lines, such that;
        % X is beginning, midpoint, end
        x = [o.prt_pos(i, 1), ...
            .5 *(o.prt_pos(i, 1)+o.prt_pos(j, 1)), ...
            o.prt_pos(j, 1)]';
        % Y is 0, distance, 0
        y = [zeros(length(i), 1), ...
            (o.prt_pos(i,1)-o.prt_pos(j,1)), ...
            zeros(length(j), 1)]';
        plot(x, y, 'Color', [.5,.5,.5]);
    case 2
        % Plot boundary particles
        x = [o.prt_pos(i, 1), o.prt_pos(j, 1)]';
        y = [o.prt_pos(i, 2), o.prt_pos(j, 2)]';
        plot(x, y, 'Color', [.5 .5 .5]);
    case 3
        % Plot boundary particles
        x = [o.prt_pos(i, 1), o.prt_pos(j, 1)]';
        y = [o.prt_pos(i, 2), o.prt_pos(j, 2)]';
        z = [o.prt_pos(i, 3), o.prt_pos(j, 3)]';
        plot3(x, y, z, 'Color', [.5 .5 .5]);
    otherwise
        hold('off');
        error('Must be in [1,3] dimensions!');
end
hold('off');

% Scale image
axis('image');
if o.par_dim >= 1; xlim([o.geo_min(1), o.geo_max(1)]); end
if o.par_dim >= 2; ylim([o.geo_min(2), o.geo_max(2)]); end
if o.par_dim >= 3; zlim([o.geo_min(3), o.geo_max(3)]); end

% Set figure background color
set(gca,'Color',[0,0,0]);

end
