function drawParticles(o)
%DRAWPARTICLES Draw particles on the current figure
clf;

hold('off');
switch o.dim
    case 1
        % Plot boundary particles
        sf = scatter(o.bound.r(:,1), zeros(o.bNum, 1), ...
            20, .75*[1,1,1], 'filled', 'hexagram');
        % Plot the actual particles
        hold('on');
        sb = scatter(o.fluid.r(:,1), zeros(o.fNum, 1), ...
            10, o.memColors(o.pNum), 'filled');
    case 2
        % Plot boundary particles
        sf = scatter(o.bound.r(:,1), o.bound.r(:,2), ...
            20, .75*[1,1,1], 'filled', 'hexagram');
        % Plot the actual particles
        hold('on');
        sb = scatter(o.fluid.r(:,1), o.fluid.r(:,2), ...
            10, o.memColors(o.fNum), 'filled');
    case 3
        % Plot boundary particles
        sf = scatter3(o.fluid.r(:,1), o.fluid.r(:,2), o.fluid.r(:,3), ...
            20, .75*[1,1,1], 'filled', 'hexagram');
        % Plot the actual particles
        hold('on');
        sb = scatter3(o.bound.r(:,1), o.bound.r(:,2), o.bound.r(:,3), ...
            10, o.memColors(o.fNum), 'filled');
    otherwise
        error('Must be in [1,3] dimensions!');
end
hold('off');

% Scale image
axis('image');
% Set figure background color
set(gca,'Color',[0,0,0]);

end
