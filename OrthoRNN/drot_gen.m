function [ G ] = drot_gen(theta, pair, N)
%so_gen: Creates the derivative dG^{ij}/dtheta for SO(N)
%   Upper is pos, lower is neg.

G = speye(N);
i = pair(1); j = pair(2);
if (i == j) || (i > N) || (j > N)
    error('Indices must be !=, <= N');
elseif i > j
    [i, j] = deal(j, i);
end

G(i,i) = -sin(theta);
G(j,j) = M(i,i);
G(i,j) = cos(theta);
G(j,i) = -M(i,j);

end

