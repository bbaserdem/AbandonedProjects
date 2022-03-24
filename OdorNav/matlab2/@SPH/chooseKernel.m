function chooseKernel(o, KER)
%CHOOSEKERNEL Hard code the kernel function

switch KER
    case {'wendland', 'Wendland'}
        o.geo_int = 2 * o.geo_smt;
        o.geo_ker = 'wendland';
        o.int_kf = @(x) math.wendland(x/o.geo_smt, o.par_dim) / ...
            (o.geo_smt^o.par_dim);
        o.int_kd = @(x) math.wendland_deriv(x/o.geo_smt, o.par_dim) / ...
            (o.geo_smt^(o.par_dim+1));
    case {'quintic','Quintic'}
        o.geo_int = 3 * o.geo_smt;
        o.geo_ker = 'quintic';
        o.int_kf = @(x) math.quintic(x/o.geo_smt, o.par_dim) / ...
            (o.geo_smt^o.par_dim);
        o.int_kd = @(x) math.quintic_deriv(x/o.geo_smt, o.par_dim) / ...
            (o.geo_smt^(o.par_dim+1));
    otherwise
        error('Kernel type not recognized');
end

end

