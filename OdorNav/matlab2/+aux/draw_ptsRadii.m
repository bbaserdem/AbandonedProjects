function draw_ptsRadii(cor, rad)
%DRAW_PTSRADII Draw points as a scatter plot, and draw individual cirles
%around them

N = size(cor, 1);
if rad > 0
    COL = aux.get_rainbow(N+1);
    COL(end,:) = [];
else
    COL = .5 * [1 1 1];
end

% Draw points
scatter( cor(:,1), cor(:,2), 100, COL, 'filled' );

% Draw the cirles
if rad > 0
    hold('on');
    for n = 1:N
        viscircles(cor(n,:), rad, 'Color', COL(n,:));
    end
    hold('off');
end

end

