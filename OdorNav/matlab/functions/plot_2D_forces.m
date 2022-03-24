function plot_2D_forces(POS, BDR, FOR)
%PLOT_2D_VELOCITY Plot the particles of the SPH formalism

% Memoize color function
if false
    persistent colors
    if isempty(colors)
        colors = memoize(@(u) get_rainbow(u, .75));
    end
else
    colors = @(u) get_rainbow(u, .75);
end

% Prepare figure handle
persistent figureHandle
if isempty(figureHandle)
    figureHandle = figure(11);
else
    figure(figureHandle);
end
clf(figureHandle);
hold('on');

% Plot boundary particles
scatter(BDR(:,1), BDR(:,2), 100, .75*[1,1,1], 'filled', 'hexagram');

% Plot the actual particles
scatter(POS(:,1), POS(:,2), 100, colors(size(POS,1)), 'filled');

% Draw forces
if exist('FOR', 'var')
    % Total
    p(8) = quiver( POS(:,1), POS(:,2), FOR.total(:,1), FOR.total(:,2), ...
        'w', 'LineWidth', 3);
    % Particle forces
%     p(4) = quiver( POS(:,1), POS(:,2), FOR.p_back(:,1), FOR.p_back(:,2) );
%     p(3) = quiver( POS(:,1), POS(:,2), FOR.p_visc(:,1), FOR.p_visc(:,2) );
%     p(2) = quiver( POS(:,1), POS(:,2), FOR.p_drif(:,1), FOR.p_drif(:,2) );
%     p(1) = quiver( POS(:,1), POS(:,2), FOR.p_pres(:,1), FOR.p_pres(:,2) );
    % Boundary forces
    p(7) = quiver( POS(:,1), POS(:,2), FOR.b_wall(:,1), FOR.b_wall(:,2) );
    p(6) = quiver( POS(:,1), POS(:,2), FOR.b_visc(:,1), FOR.b_visc(:,2) );
    p(5) = quiver( POS(:,1), POS(:,2), FOR.b_pres(:,1), FOR.b_pres(:,2) );
    legend(p(5:end), ...
        'Pressure (Boun)', 'Viscous (Boun)', 'Repulsion (Boun)', ...
        'Total' );
%         'Pres. (Part)', 'Visc. (Part)', 'Drift (Part)', 'Bckg (Part)', ...
end


% Scale image
axis('image');
% Set figure background color
set(gca,'Color',[0,0,0]);
hold('off');

end