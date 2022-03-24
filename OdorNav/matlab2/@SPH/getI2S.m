function out = getI2S(shp, ind)
% GETI2S Convert vector of linear index to subscript
out = zeros(size(ind,1),length(shp));
ind = ind-1;
for d = 1:length(shp)
    out(:,d) = mod(ind,shp(d))+1;
    ind = (ind-out(:,d)+1)/shp(d);
end
end