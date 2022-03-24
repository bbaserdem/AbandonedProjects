function [I, X] = getAbsPartitions(A)
%GETABSPARTITIONS Partition label space in abstract labels
% I: Cell, with labels corresponding

% Cast L as uint64, and get nonzeros, along with mapping to original index(Q)
Q = uint64(find(A>0));
L = uint64(A(Q));

% Sort the non-zero labels and get the locations of state transitions
[V,N] = sort(L);
T = find( V(2:end) - V(1:(end-1)) );
T = [uint64(0); T; uint64(length(N))];

% Fix new labels
F = Q(N);

% Get the label values which exist
X = V(T(2:end));

% Create the cell array
I = cell(length(X), 1);

% For every label value(X(i)) 
for i = 1:length(X)
    % Put the list of particles in this voxel to the cell array
    I{i} = F((T(i)+1):T(i+1))';
end

end

