function o = cell_times_mat(u, v)
%CELL_TIMES_MAT Take the times with a cell
o = cellfun(@(x) x.*u, v, 'UniformOutput', false);
end

