function drawIntFluid(o)
%DRAWINTFLUID Draw the interactions between fluid particles on plot

hold('on');
switch o.dim
    case 1
        % Draw triangle lines, such that;
        % X is beginning, midpoint, end
        x = [o.fluid.r(o.Fi_ij), ...
            .5*(o.fluid.r(o.Fi_ij)+o.fluid.r(o.Fj_ij)), ...
            o.fluid.r(o.Fj_ij)]';
        % Y is 0, distance, 0
        y = [zeros(length(o.Fi_ij),1), ...
            (o.fluid.r(o.Fi_ij)-o.fluid.r(o.Fj_ij)), ...
            zeros(length(o.Fj_ij),1)]';
        plot(x, y, 'Color', [.5,.5,.5]);
    case 2
        % Plot boundary particles
        x = [o.fluid.r(o.Fi_ij,1), o.fluid.r(o.Fj_ij,1)]';
        y = [o.fluid.r(o.Fi_ij,2), o.fluid.r(o.Fj_ij,2)]';
        plot(x, y, 'Color', [.5 .5 .5]);
    case 3
        % Plot boundary particles
        x = [o.fluid.r(o.Fi_ij,1), o.fluid.r(o.Fj_ij,1)]';
        y = [o.fluid.r(o.Fi_ij,2), o.fluid.r(o.Fj_ij,2)]';
        z = [o.fluid.r(o.Fi_ij,3), o.fluid.r(o.Fj_ij,3)]';
        plot3(x, y, z, 'Color', [.5 .5 .5]);
    otherwise
        hold('off');
        error('Must be in [1,3] dimensions!');
end
hold('off');

% Scale image
axis('image');
% Set figure background color
set(gca,'Color',[0,0,0]);

end
