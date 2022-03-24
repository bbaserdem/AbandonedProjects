function out = getAdjacent(D)
% GETADJACENT Get linear index of grid points that are adjacent in D space

% Get a row which gives the linear index displacement
S = [1, cumprod(D(1:(end-1)))];
% Generate list of positive change vectors
V = SPH.getPositiveVec(length(D));
% Calculate the row of linear index displacement for the ones in V
L = S * V';
% Generate the subscrip index for all the voxels
O = SPH.getI2S(D,(1:prod(D))');
% Calculate the linear index of all adjacents, and negative adjacents
A = ((1:prod(D))') + L;

% Null the invalid indices;
for d = 1:length(D)
    % When dim index;O(<i>,d) was max;shp(d) and tried to inc;V(<j>)== 1
    A(O(:,d)==D(d), V(:,d)>0) = 0;
    % When dim index;O(<i>,d) was min;1      and tried to dec;V(<j>)==-1
    A(O(:,d)==1,      V(:,d)<0) = 0;
end

% Get the index list
B = ((1:prod(D))') .* ones(1, size(V,1));
out = sparse(B(A(:)~=0), A(A(:)~=0), true, prod(D), prod(D));

end
