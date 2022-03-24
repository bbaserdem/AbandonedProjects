function drawVectors(o, pArr, bArr, colorize)
%DRAWPARTICLES Draw arrows on the current figure

if ~exist('colorize','var')
    colorize = false;
end

hold('on');
switch o.dim
    case 1
        % Slant the arrows a bit for visibility
        epsP = .1*(max(abs(pArr(:))) - min(abs(pArr(:))));
        q = quiver(...
            o.fluid.r, ...
            zeros(o.flNum, 1), ...
            pArr, ...
            epsP*ones(o.fNum, 1) );
        if exist('bArr','var')
            epsB = .1*(max(abs(bArr(:))) - min(abs(bArr(:))));
            quiver(...
                o.bound, zeros(o.bNum, 1), ...
                bArr, epsB*ones(o.bNum, 1) );
        end
    case 2
        q = quiver(...
            o.fluid.r(:,1), o.fluid.r(:,2), ...
            pArr(:,1), pArr(:,2), ...
            5 * o.h, 'Color', [1, 1, 1], 'LineWidth', 2 );
        if exist('bArr','var')
            quiver(...
                o.bound.r(:,1), o.bound.r(:,2), ...
                bArr(:,1), bArr(:,2), ...
                5*o.h, 'Color', [.7 .7 .7], 'LineWidth', 1.5 );
        end
    case 3
        q = quiver3(...
            o.fluid.r(:,1), o.fluid.r(:,2), o.fluid.r(:,3), ...
            pArr(:,1), pArr(:,2), pArr(:,3), ...
            'Color', [1. 1. 1.] );
        if exist('bArr','var')
            quiver(...
                o.bound.r(:,1), o.bound.r(:,2), o.bound.r(:,3), ...
                bArr(:,1), bArr(:,2), bArr(:,3), ...
                'Color', [.7 ,.7, .7] );
        end
    otherwise
        error('Must be in [1,3] dimensions!');
end
hold('off')

if colorize % Change colors
    v = sqrt(sum(pArr.^2, 2));
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

end