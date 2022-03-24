function V = get_positiveVec(D)
%GET_POSITIVEVEC Get all positive vectors in square ND grid

V = zeros(1,0);
G = zeros(1,0);

for d = 1:D
    V = [ zeros(size(V,1), 1), V; ones( size(G,1), 1), G];
    G = [ repmat( (-1:1)', size(G, 1), 1), repelem(G, 3, 1)];
end

% Remove the null vector
V(all(V==0,2),:) = [];

end

