function [dist, ehat, i, j] = getOpt_dist_pair(pos, thr)
%GETOPT_DIST_PAIR Get optimized distances within range
%   Algorithm is;
%       * Get hashes (linear-index)
%       * Get adjacency between hashes and index (getOpt_adjacency)
%       * Check adjacency

[NUM, DIM] = size(pos);

% Initialize pairwise index function that is memoized
persistent pairindex
if isempty(pairindex)
    pairindex = memoize(@get_i_gt_j);
end
[I, J] = pairindex(NUM);

% Initialize adjacency function that is memoized
persistent adjacency
if isempty(adjacency)
    adjacency = memoize(@getOpt_adjacency);
end

% Hash
s2i = @(s, z) 1 + sum( [1,cumprod(s(1:(end-1)))] .* (z-1), 2 );
% Create hash list, from index ids of [1,...,1] to [M,...,N]
subHash = floor(pos/thr);
subHash = hash-min(hash, [], 1)+1;
dimHash = max(subHash, [], 1);
linHash = s2i(dimHash, subHash);
adjHash = adjacency(dimHash);


% Faster index functions
end

