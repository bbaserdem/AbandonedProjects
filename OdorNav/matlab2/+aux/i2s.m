function sub = i2s(shp, ind)
% I2S Convert vector of linear index to subscript
sub = zeros(size(ind,1),length(shp));
ind = ind-1;
for d = 1:length(shp)
    sub(:,d) = mod(ind,shp(d))+1;
    ind = (ind-sub(:,d)+1)/shp(d);
end
end