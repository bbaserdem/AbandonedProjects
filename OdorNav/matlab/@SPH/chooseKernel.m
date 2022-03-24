function chooseKernel(o, KER)
%CHOOSEKERNEL Hard code the kernel function

switch KER
    case {'wendland', 'Wendland'}
        o.support = 2 * o.smoothing;
        o.kernel = 'wendland';
        o.fn_ker = @(x) math.wendland(x, o.dimension) / ...
            (o.smoothing^o.dimension);
        o.fn_kdr = @(x) math.wendland_deriv(x, o.dimension) / ...
            (o.smoothing^(o.dimension+1));
    case {'quintic','Quintic'}
        o.support = 3 * o.smoothing;
        o.kernel = 'quintic';
        o.fn_ker = @(x) math.quintic(x, o.dimension) / ...
            (o.smoothing^o.dimension);
        o.fn_kdr = @(x) math.quintic_deriv(x, o.dimension) / ...
            (o.smoothing^(o.dimension+1));
    otherwise
        error('Kernel type not recognized');
end

end

