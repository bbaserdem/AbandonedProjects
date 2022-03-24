function [dist, ehat, i, j] = get_dist_pair(pos, thr)
%GET_DIST_PAIR Get the distances between the points

[NUM, DIM] = size(pos);

% Initialize pairwise index function that is memoized
persistent pairindex
if isempty(pairindex)
    pairindex = memoize(@get_i_gt_j);
end
[I, J] = pairindex(NUM);

VEC = pos(I,:) - pos(J,:);
DIS = zeros(size(VEC, 1), 1);

% Eliminate candidates out of grid
sto = all(abs(VEC) < thr, 2);
% Check the distances for all the candidates
DIS(sto) = sum(VEC(sto, :).^2, 2);
% Trim the offenders
sto(sto) = (DIS(sto) < (thr.^2)) & (DIS(sto) > 0);

i = I(sto);
j = J(sto);
v = sqrt(DIS(sto));

dist = sparse([i; j], [j; i], [v; v], NUM, NUM);
r = VEC(sto, :) ./ v;
ehat = cell(DIM,1);
for d = 1:DIM
    ehat{d} = sparse([i; j], [j; i], [r(:, d); -1*r(:, d)], NUM, NUM);
end

end