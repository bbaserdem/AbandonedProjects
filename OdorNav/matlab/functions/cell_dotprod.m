function o = cell_dotprod(u, v)
%CELL_DOTPROD Take the dot product accross the cell index
sto = cellfun(@times, u, v, 'UniformOutput', false);
o = spalloc(size(sto{1},1), size(sto{1},2), nnz(sto{1}));
for d = length(sto)
    o = o + sto{d};
end

end
