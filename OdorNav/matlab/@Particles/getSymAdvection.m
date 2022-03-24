function A = getSymAdvection(o, i, j, b)
%GETSYMADVECTION Get the SparseTensor matrix of the advection tensor;
%   A_i = rho_i u_i (*) ( v_i - u_i )
%   If another object b is provided, it is calculated among the two

ao = (o.d.*o.u) .* reshape(o.v-o.u, o.num, 1, o.dim);
if ~exist('b', 'var')
    % If no other object is given, calculate on self;
    a = .5*(ao(i,:,:)+ao(j,:,:));
    A = SparseTensor([i;j], [j;i], [a;a], o.num, o.num);
elseif isobject(b)
    ab = (b.d.*b.u) .* reshape(b.v-b.u, b.num, 1, b.dim);
    a = .5*(ao(i,:,:)+ab(j,:,:));
    A = SparseTensor(i, j, a, o.num, b.num);
elseif iscalar(b)
    a = .5*ao(i,:,:);
    A = SparseTensor(i, j, a, o.num, b);
else
    error('Last argument must be a valid object');
end

end

