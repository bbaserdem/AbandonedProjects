function o = cell_contract(u)
%CELL_CONTRACT Contract a cell.
sto = cellfun(@(x) full(sum(x, 2)), u, 'UniformOutput', false);
o = cell2mat(sto');
end

