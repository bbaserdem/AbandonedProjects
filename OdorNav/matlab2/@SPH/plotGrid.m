function plotGrid(o)
%PLOTGRID Arrange the grid of the plot to reflect the hashing

if o.par_dim >= 1
    xticks(((2:o.hsh_range(1))+o.hsh_offset(1))*o.geo_int);
end
if o.par_dim >= 2
    yticks(((2:o.hsh_range(2))+o.hsh_offset(2))*o.geo_int);
end
if o.par_dim >= 3
    zticks(((2:o.hsh_range(3))+o.hsh_offset(3))*o.geo_int);
end

grid('on');
a = gca;
a.GridColor = ones(1,3);

% Scale image
axis('image');
if o.par_dim >= 1; xlim([o.geo_min(1), o.geo_max(1)]); end
if o.par_dim >= 2; ylim([o.geo_min(2), o.geo_max(2)]); end
if o.par_dim >= 3; zlim([o.geo_min(3), o.geo_max(3)]); end

end

