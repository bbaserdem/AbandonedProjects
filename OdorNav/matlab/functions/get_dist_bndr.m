function [dist, ehat, i, j] = get_dist_bndr(pos, bdr, thr)
%GET_DIST_BNDR Get the distances between two sets of points

% Initialize pairwise index function that is memoized
[NUM, DIM] = size(pos);
BOR = size(bdr, 1);

% Initialize pairwise index function that is memoized
persistent ijindex
if isempty(ijindex)
    ijindex = memoize(@get_i_j);
end
[I, J] = ijindex(NUM, BOR);

% Get distance vector
VEC = pos(I,:) - bdr(J,:);
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
n = sum(sto);

dist = sparse(i, j, v, NUM, BOR);
r = VEC(sto,:) ./ v;
ehat = cell(DIM,1);
for d = 1:DIM
    ehat{d} = sparse(i, j, r(:, d), NUM, BOR);
end

end