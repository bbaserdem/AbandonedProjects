function plotParticles(o)
%PLOTPARTICLES Draw particles on the current figure

% Clear this figure
cla;

% Fluid
fluid_color = o.getColors(o.prt_fld_num);
fluid_coord = o.prt_pos_fld;

% Boundary particles
bound_color = .75 * [1, 1, 1];
bound_coord = o.prt_pos_bdr;

% Inlet particles
inlet_color = 1 * [.8, 1, .8];
inlet_coord = o.prt_pos_inl;

% Outlet particles
outlet_color = 1 * [1, .8, .8];
outlet_coord = o.prt_pos_bdr;

hold('on');
switch o.par_dim
    case 1
        % Plot the fluid particles
        scatter( ...
            fluid_coord(:,1), zeros(length(fluid_coord), 1), ...
            20, fluid_color, 'filled');
        % Plot the boundary particles
        scatter( ...
            bound_coord(:,1), zeros(length(bound_coord), 1), ...
            20, bound_color, 'filled', 'hexagram');
        % Plot the inlet particles
        scatter( ...
            inlet_coord(:,1), zeros(length(inlet_coord), 1), ...
            20, inlet_color, 'filled', '+');
        % Plot the outlet particles
        scatter( ...
            outlet_coord(:,1), zeros(length(outlet_coord), 1), ...
            20, outlet_color, 'filled', 'o');
    case 2
        % Plot the fluid particles
        scatter( ...
            fluid_coord(:,1), fluid_coord(:,2), ...
            20, fluid_color, 'filled');
        % Plot the boundary particles
        scatter( ...
            bound_coord(:,1), bound_coord(:,2), ...
            20, bound_color, 'filled', 'hexagram');
        % Plot the inlet particles
        scatter( ...
            inlet_coord(:,1), inlet_coord(:,2), ...
            20, inlet_color, 'filled', '+');
        % Plot the outlet particles
        scatter( ...
            outlet_coord(:,1), outlet_coord(:,2), ...
            20, outlet_color, 'filled', 'o');
    case 3
        % Plot the fluid particles
        scatter( ...
            fluid_coord(:,1), fluid_coord(:,2), fluid_coord(:,3), ...
            20, fluid_color, 'filled');
        % Plot the boundary particles
        scatter( ...
            bound_coord(:,1), bound_coord(:,2), bound_coord(:,3), ...
            20, bound_color, 'filled', 'hexagram');
        % Plot the inlet particles
        scatter( ...
            inlet_coord(:,1), inlet_coord(:,2), inlet_coord(:,3), ...
            20, inlet_color, 'filled', '+');
        % Plot the outlet particles
        scatter( ...
            outlet_coord(:,1), outlet_coord(:,2), outlet_coord(:,3), ...
            20, outlet_color, 'filled', 'o');
    otherwise
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
