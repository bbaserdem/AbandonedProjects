function plot_2D_velocity(POS, BDR, VEL, PCN, BCN)
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
    figureHandle = figure(10);
else
    figure(figureHandle);
end
clf(figureHandle);
hold('on');

% Draw connections,
if exist('BCN', 'var')
    plot(...
        [POS(BCN(:, 1), 1), BDR(BCN(:, 2), 1)]', ...
        [POS(BCN(:, 1), 2), BDR(BCN(:, 2), 2)]', ...
        'Color', .15*ones(1,3));
end
if exist('PCN', 'var')
    plot(...
        [POS(PCN(:, 1), 1), POS(PCN(:, 2), 1)]', ...
        [POS(PCN(:, 1), 2), POS(PCN(:, 2), 2)]', ...
        'Color', .5*ones(1,3));
end

% Plot boundary particles
scatter(BDR(:,1), BDR(:,2), 100, .75*[1,1,1], 'filled', 'hexagram');

% Plot the actual particles
scatter(POS(:,1), POS(:,2), 100, colors(size(POS,1)), 'filled');

% Draw velocities
if exist('VEL', 'var')
    q = quiver( POS(:,1), POS(:,2), VEL(:,1), VEL(:,2), ...
        'Color', 'w' );
    % Change colors
    v = sqrt(sum(VEL.^2, 2));
    v = (v-min(v(:)))/(1e-5+max(v(:))-min(v(:)));
    col = colormap(gca);
    
    %// Now determine the color to make each arrow using a colormap
    [~, ~, ind] = histcounts(v, size(col, 1));
    
    %// Now map this to a colormap to get RGB
    cmap = uint8(ind2rgb(ind(:), col)*255);
    cmap(:,:,4) = 255;
    cmap = permute(repmat(cmap, [1,3,1]), [2,1,3]);
    
    %// We repeat each color 3 times (using 1:3 below) because each arrow has 3 vertices
    set(q.Head, ...
        'ColorBinding', 'interpolated', ...
        'ColorData', reshape(cmap(1:3,:,:), [], 4).');
    
    %// We repeat each color 2 times (using 1:2 below) because each tail has 2 vertices
    set(q.Tail, ...
        'ColorBinding', 'interpolated', ...
        'ColorData', reshape(cmap(1:2,:,:), [], 4).');
end


% Scale image
axis('image');
% Set figure background color
set(gca,'Color',[0,0,0]);
hold('off');

end