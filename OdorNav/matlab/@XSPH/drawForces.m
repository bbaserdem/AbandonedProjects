function drawForces(o, K)
%DRAWFORCES Draw the forces on the current figure

if ~exist('K','var')
    K = 'all';
end

% Force types
f_names = {...
    'Pressure'; 'Advection'; 'Viscous'; 'External'; 'Background'; 'Total'};
% Consolidate to single matrix
f_all = cat(3, o.F_pre, o.F_adv, o.F_vis, o.F_ext, o.F_bkg);
f_all = cat(3, f_all, sum(f_all,3));
% Number of quiver plots to draw
f_n = size(f_all,3);
% Color of quivers
f_color = hsv2rgb([linspace(1/f_n,1,f_n); ones(2,f_n)]');
% Only show requested forces
mask = false(f_n,1);
if ischar(K)
    K = {K};
end
if iscell(K)
    for i = 1:length(K(:))
        switch K{i}
            case {'pre', 'pressure', 'Pressure'}
                mask(1) = true;
            case {'adv', 'advection', 'Advection'}
                mask(2) = true;
            case {'vis', 'viscous', 'Viscous'}
                mask(3) = true;
            case {'ext', 'external', 'External'}
                mask(4) = true;
            case {'bkg', 'background', 'Background'}
                mask(5) = true;
            case {'tot', 'total', 'Total'}
                mask(end) = true;
            case {'all', 'All'}
                mask(:) = true;
            otherwise
                error('Unknown option');
        end
    end
elseif isnumeric(K)
    mask(K) = true;
elseif islogical(K)
    mask = K;
else
    error('Unknown specifier');
end
% Rearrange accordingly
f_n = sum(mask);
f_names = f_names(mask);
f_all = f_all(:,:,mask);
f_color = f_color(mask,:);

% PLOTTING
hold('on');
q = gobjects(f_n, 1);
switch o.dim
    case 1
        % Slant the arrows a bit for visibility
        eps = .1*range(f_all,'all')*ones(o.fNum,f_n).*linspace(-1,1,f_n);
        for i = 1:f_n
            q(i) = quiver(...
            o.fluid.r(:,1), zeros(o.fNum, 1), ...
            f_all(:,1,i), eps(:,i), ...
            'Color', f_color(i,:), 'LineWidth', 2 );
        end
    case 2
        for i = 1:f_n
            q(i) = quiver(...
            o.fluid.r(:,1), o.fluid.r(:,2), ...
            f_all(:,1,i), f_all(:,2,i), ...
            'Color', f_color(i,:), 'LineWidth', 2 );
        end
    case 3
        for i = 1:f_n
            q(i) = quiver3(...
            o.fluid.r(:,1), o.fluid.r(:,2), o.fluid.r(:,3), ...
            f_all(:,1,i), f_all(:,2,i), f_all(:,3,i), ...
            'Color', f_color(i,:), 'LineWidth', 2 );
        end
    otherwise
        error('Must be in [1,3] dimensions!');
end
hold('off')
legend(q, cellfun(@(x)['\color{white} ',x],f_names,'UniformOutput',false));

% Scale image
axis('image');
% Set figure background color
set(gca,'Color',[0,0,0]);

end