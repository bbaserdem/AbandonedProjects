function calculateDensity(o)
%CALCULATEDENSITY Calculates density and pressure of particles

% Calculate density of fluid particle
o.fluid.d = o.fluid.m .* full( ...
    sum(o.Fw_ij, 2) + ...
    sum(o.Bw_ij, 2) );

% Calculate volume of fluid particle
o.fluid.s = o.fluid.m ./ o.fluid.d;

% Calculate pressure of fluid particle
o.fluid.p = o.pressure(o.fluid.d);

end

