function [I, J] = get_i_j(N,M)
%GET_I_J Get list of particle indices that are i=1:N, j=1:M 

% Initialize; there are N*(N-1)/2 matches
I = repmat( (1:N)', M, 1);
J = repelem((1:M)', N, 1);

end

