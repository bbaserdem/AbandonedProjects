function V = getDiffVelocity(o, i, j, b)
%GETDIFFVELOCITY Get the SparseTensor matrix of pressure, symmetrized
%   If another object b is provided, it is calculated among the two

if ~exist('b', 'var')
    % If no other object is given, calculate on self;
    v = o.u(i,:) - o.u(j,:);
    V = SparseTensor([i;j], [j;i], [v;-v], o.num, o.num);
elseif isobject(b)
    % If no other object is given, calculate on self;
    v = o.u(i,:) - b.u(j,:);
    V = SparseTensor(i, j, v, o.num, b.num);
else
    error('Last argument must be a valid object');
end

end

