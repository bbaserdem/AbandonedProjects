function N = getSymFriction(o, i, j, b)
%GETSYMFRICTION Get the sparse matrix of pressure, symmetrized
%   If another object b is provided, it is calculated among the two
EPS = 1e-20;

if ~exist('b', 'var')
    % If no other object is given, calculate on self;
    n = 2 * (o.n(i).*o.d(i).*o.n(j).*o.d(j)) ./ ( ...
        (o.n(i).*o.d(i)) + (o.n(j).*o.d(j)) );
    N = sparse([i;j], [j;i], [n;n], o.num, o.num);
elseif isobject(b)
    n = 2 * (o.n(i).*o.d(i).*b.n(j).*b.d(j)) ./ ( ...
        (o.n(i).*o.d(i)) + (b.n(j).*b.d(j)) );
    N = sparse(i, j, n, o.num, b.num);
else
    error('Last argument must be a valid object');
end

end

