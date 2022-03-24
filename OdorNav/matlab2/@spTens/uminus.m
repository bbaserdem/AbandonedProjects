function oo = uminus(o)
%UMINUS Overload unary minus function

oo = SparseTensor(cellfun(@uminus, o.data, 'UniformOutput', false));

end

