function [uR, L, I, J] = unique_list(R)

% UNIQUE_LIST :
% [UM, L] = unique_list(M)
% M - matrix
% UM - unique rows
% L - list of rows in M for each UM
%

[uR, I, J] = unique(R, 'rows');
L = accumarray(J, [1:length(J)]', [], @(x) {x});


end

