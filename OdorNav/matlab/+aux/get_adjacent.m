function I = get_adjacent(shp)
% GETADJACENT Get linear index of grid points that are adjacent in D space
%   Given a shape, (of L=prod(shp) points in D=size(shp,1) space) :
%       A=(L,3^D) where the row A(i,:) contains linear indices which are
%       adjacent to the i'th linear index in D-space

% Get a row which gives the linear index displacement
S = [1, cumprod(shp(1:(end-1)))];
% Generate list of positive change vectors
V = aux.get_positiveVec(length(shp));
% Calculate the row of linear index displacement for the ones in V
L = S * V';
% Generate the subscrip index for all the voxels
O = aux.i2s(shp,(1:prod(shp))');
% Calculate the linear index of all adjacents, and negative adjacents
A = ((1:prod(shp))') + L;

% Null the invalid indices;
for d = 1:length(shp)
    % When dim index;O(<i>,d) was max;shp(d) and tried to inc;V(<j>)== 1
    A(O(:,d)==shp(d), V(:,d)>0) = 0;
    % When dim index;O(<i>,d) was min;1      and tried to dec;V(<j>)==-1
    A(O(:,d)==1,      V(:,d)<0) = 0;
end

% Get index list
B = ((1:prod(shp))') .* ones(1, size(V,1));
% Put into sparse boolean
I = sparse(B(A(:)~=0), A(A(:)~=0), true, prod(shp), prod(shp));

end
