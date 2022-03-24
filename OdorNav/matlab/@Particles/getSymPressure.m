function P = getSymPressure(o, i, j, b)
%GETSYMPRESSURE Get the sparse matrix of pressure, symmetrized
%   If another object b is provided, it is calculated among the two

if ~exist('b', 'var')
    % If no other object is given, calculate on self;
    p = ((o.p(i).*o.d(j))+(o.p(j).*o.d(i)))./(o.d(i)+o.d(j));
    P = sparse([i;j], [j;i], [p;p], o.num, o.num);
elseif isobject(b)
    p = ((o.p(i).*b.d(j))+(b.p(j).*o.d(i)))./(o.d(i)+b.d(j));
    P = sparse(i, j, p, o.num, b.num);
else
    error('Last argument must be a valid object');
end

end

