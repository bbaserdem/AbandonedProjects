function oo = dot(o, b)
%DOT Tensorial dot product
%   * If it is two SparseMatrix, the matrices are matrix multiplied on the
%   tensor index, and elementwise multiplied on the sparse index
%   * If a SparseMatrix is multiplied with a numeric array, each element is
%   matrix multiplied instead.

if isobject(o) && isnumeric(b)
    
else
    error('Not configured');
end

% If the end result is a scalar array, just return the sparse matrix
if isscalar(d)
    oo = d{1};
else
    oo = SparseTensor(d);
end

end

function S = sum_cell(C)
%SUM_CELL sum together all cell array
S = C{1};
for i = 2:numel(C)
    S = S + C{i};
end
end