function [I, J] = get_ij(N,M)
%GET_IJ Get list of particle indices
%   If two arguments, then list is i=1:N, j=1:M
%   If single argument, then i=1:N, j=1:N|j<i

if exist('M','var')
    I = repmat( (1:N)', M, 1);
    J = repelem((1:M)', N, 1);
else
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

end

