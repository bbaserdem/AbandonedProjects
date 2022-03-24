function oo = mtimes(o, b)
%MTIMES Matrix multiplication (tensor contraction with first and last ind)
%   * If it is two SparseMatrix, the matrices are matrix multiplied on the
%   tensor index, and elementwise multiplied on the sparse index
%   * If a SparseMatrix is multiplied with a numeric array, each element is
%   matrix multiplied instead.

if isobject(b) && isobject(o)
    % Check dims
    if o.dim(end) ~= b.dim(1)
        error('Mismatching dimensions between the two SparseTensor''s');
    else
        X = b.dim(1);
    end
    
    oC = reshape(o.data,[], X);
    bC = reshape(b.data, X,[]);
    d = cell(size(oC,1), size(bC,2));
    for i = 1:size(d,1)
        for j = 1:size(d,2)
            d{i,j} = sum_cell(...
                cellfun(@times, oC(i,:), bC(:,j)', 'UniformOutput', false));
        end
    end
    d = reshape(d, [o.dim(1:(end-1)), b.dim(2:end)]);
elseif isobject(b)
    d = cellfun(@(x) sparse(o*x), b.data, 'UniformOutput', false);
else
    d = cellfun(@(x) sparse(x*b), o.data, 'UniformOutput', false);
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