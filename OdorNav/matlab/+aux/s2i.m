function ind = s2i(shp, sub)
% S2I Convert subscript matrix to linear index
ind = 1 + sum( [1,cumprod(shp(1:(end-1)))] .* (sub-1), 2 );
end