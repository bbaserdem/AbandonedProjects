function A = getOpt_adjacency(S)
%GETOPT_ADJACENCY Get adjacent linear indices from dimensions S

% Fast subscript to linear index calculator
s2i = @(s, z) 1 + sum( [1,cumprod(s(1:(end-1)))] .* (z-1), 2 );
i2s = @(s, z) 1 + [ mod(z-1,R), floor((z-1)/R) ];

% Generate matrix of indices
I = i2s(S, (1:prod(S))');

% Generate delta index vectors
for d = 1:d
    D = 

% Pre-allocate A
A = zeros(S, 3^length(S));
% Generate A
for d = 1:length(S)
    for del = -1:0:1

% Generate offset vectors

end

