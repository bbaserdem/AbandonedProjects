function oo = transpose(o)
%TRANSPOSE Overload transpose function
%   Use the non-complex transpose (o.') to transpose the sparse indices

oo = SparseTensor(cellfun(@transpose, o.data, 'UniformOutput', false));

end

