function [I, R, K] = get_partitions(A,M)
%GET_PARTITIONS Partition a label list, labels running from 1 to M (0 ignored)
%   I: {i} Cell of index with label;                    A(I{i}) = i
%   R: (n,1) Reverse lookup list that gives location;   I{i}(R(n))==n
%   K: (i,n) Boolean label matrix,                      K(i,n) = A(n)==i

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

% Recast the label max (M) if not defined
if ~exist('M','var')
    M = max(L(:));
end

% Create the cell array
I = cell(M, 1);
% Create the bool array
K = spalloc(M, size(A, 1), size(A, 1)) > 0;
% Create reverse lookup array
R = zeros(size(A, 1), 1);

% For every label value(X(i)) 
for i = 1:length(X)
    % Put the list of particles in this voxel to the cell array
    I{X(i)} = F((T(i)+1):T(i+1))';
    % Flip the boolean value to true on the particles
    K(i, F((T(i)+1):T(i+1))) = true;
    % Put the reverse lookup values for particles in this voxel
    R(I{X(i)}) = 1:length(I{X(i)});
end

% Create boolean lookup table if wanted
if nargout == 3
    % Create reverse lookup array
    R = zeros(size(A, 1), 1);
    % For every label value(X(i))
    for i = 1:length(X)
        % Put the reverse lookup values for particles in this voxel
        R(I{X(i)}) = 1:length(I{X(i)});
    end
    
end

end

