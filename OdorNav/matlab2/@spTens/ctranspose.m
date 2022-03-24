function oo = ctranspose(o)
%CTRANSPOSE Overload complex transpose function;
%   Use regular tick (o') to transpose the tensor index

oo = SparseTensor(o.data');

end

