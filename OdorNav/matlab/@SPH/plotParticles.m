function plotParticles(o)
%PLOTPARTICLES Draw particles on the current figure
clf;

colors = aux.get_rainbow(o.fluidNo);
hold('off');
switch o.dimension
    case 1
        % Plot boundary particles
        scatter(o.cor(o.f_bound,1), zeros(o.boundNo, 1), ...
            20, .75*[1,1,1], 'filled', 'hexagram');
        % Plot the actual particles
        hold('on');
        scatter(o.cor(o.fluid,1), zeros(o.fluidNo, 1), ...
            10, colors, 'filled');
    case 2
        % Plot boundary particles
        scatter( o.cor(o.f_bound,1), o.cor(o.f_bound,2), ...
            20, .75*[1,1,1], 'filled', 'hexagram');
        % Plot the actual particles
        hold('on');
        scatter( o.cor(o.f_fluid,1), o.cor(o.f_fluid,2), ...
            10, colors, 'filled');
    case 3
        % Plot boundary particles
        scatter3(  o.cor(o.f_bound,1), o.cor(o.f_bound,2), o.cor(o.f_bound,3), ...
            20, .75*[1,1,1], 'filled', 'hexagram');
        % Plot the actual particles
        hold('on');
        scatter3( o.cor(o.f_fluid,1), o.cor(o.f_fluid,2), o.cor(o.f_fluid,3), ...
            10, colors, 'filled');
    otherwise
        error('Must be in [1,3] dimensions!');
end
hold('off');

% Scale image
axis('image');
% Set figure background color
set(gca,'Color',[0,0,0]);

end
