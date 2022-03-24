function chooseKernel(o, KER)
%CHOOSEKERNEL Hard code the kernel function

switch KER
    case {'wendland', 'Wendland'}
        o.s = 2;
        o.kernel = @(x) math.wendland(x, o.dim) / (o.h^o.dim);
        o.kerDer = @(x) math.wendland_deriv(x, o.dim) / (o.h^(o.dim+1));
    case {'quintic','Quintic'}
        o.s = 3;
        o.kernel = @(x) math.quintic(x, o.dim) / (o.h^o.dim);
        o.kerDer = @(x) math.quintic_deriv(x, o.dim) / (o.h^(o.dim+1));
    otherwise
        error('Kernel type not recognized');
end

end

