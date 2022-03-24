function drawIntBorder(o)
%DRAWINTBORDER Draw the interactions from boundary to fluid particles

hold('on');
switch o.dim
    case 1
        % Draw triangle lines, such that;
        % X is beginning, midpoint, end
        x = [o.fluid.r(o.Bi_ij), ...
            .5*(o.fluid.r(o.Bi_ij)+o.bound.r(o.Bj_ij)), ...
            o.bound.r(o.Bj_ij)]';
        % Y is 0, distance, 0
        y = [zeros(length(o.Bi_ij),1), ...
            (o.fluid.r(o.Bi_ij)-o.bound.r(o.Bj_ij)), ...
            zeros(length(o.Bj_ij),1)]';
        plot(x, y, ':', 'Color', [.25,.25,.25]);
    case 2
        % Plot boundary particles
        x = [o.fluid.r(o.Bi_ij,1), o.bound.r(o.Bj_ij,1)]';
        y = [o.fluid.r(o.Bi_ij,2), o.bound.r(o.Bj_ij,2)]';
        plot(x, y, ':', 'Color', [.25 .25 .25]);
    case 3
        % Plot boundary particles
        x = [o.fluid.r(o.Bi_ij,1), o.bound.r(o.Bj_ij,1)]';
        y = [o.fluid.r(o.Bi_ij,2), o.bound.r(o.Bj_ij,2)]';
        z = [o.fluid.r(o.Bi_ij,3), o.bound.r(o.Bj_ij,3)]';
        plot3(x, y, z, ':', 'Color', [.25 .25 .25]);
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
