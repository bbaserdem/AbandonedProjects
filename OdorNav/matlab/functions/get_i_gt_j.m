function [I, J] = get_i_gt_j(N)
%GET_I_GT_J Get list of particle indices such that i>j 

% Initialize; there are N*(N-1)/2 matches
I = zeros( N*(N-1)/2, 1 );
J = zeros( N*(N-1)/2, 1 );
for i = 1:N
    % The previous have filled up to (i-2)*(i-1)/2
    % i-1 more will be filled
    I( ((i-2)*(i-1)/2)+(1:(i-1)) ) = i;
    J( ((i-2)*(i-1)/2)+(1:(i-1)) ) = ((1:(i-1))');
end

end

