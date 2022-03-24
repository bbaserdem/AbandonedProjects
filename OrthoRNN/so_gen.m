function [ M ] = so_gen(pair, N)
%so_gen: Creates the generator M^{ij} for SO(N)
%   Upper is pos, lower is neg.

M = zeros(N);
i = pair(1); j = pair(2);
if (i == j) || (i > N) || (j > N)
    error('Indices must be !=, <= N');
elseif i > j
    [i, j] = deal(j, i);
end

M(i,j) = 1; M(j,i) = -1;

end

