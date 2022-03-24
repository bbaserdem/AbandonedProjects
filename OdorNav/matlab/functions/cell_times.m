function o = cell_times(u, v)
%CELL_TIMES Take the times accross the cell index
o = cellfun(@times, u, v, 'UniformOutput', false);
end

