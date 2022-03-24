function out = getS2I(shp, sub)
% GETS2I Convert subscript matrix to linear index
out = 1 + sum( [1,cumprod(shp(1:(end-1)))] .* (sub-1), 2 );
end