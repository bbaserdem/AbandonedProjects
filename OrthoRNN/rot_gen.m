function [ G ] = rot_gen(theta, pair, N)
%so_gen: Creates the generator G^{ij}=e^theta.M^{ij} for SO(N)
%   Upper is pos, lower is neg.

G = speye(N);
i = pair(1); j = pair(2);

G(i,i) = cos(theta);
G(j,j) = M(i,i);
G(i,j) = sin(theta);
G(j,i) = -M(i,j);

end

