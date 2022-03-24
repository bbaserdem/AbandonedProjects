function doTimeStep(o)
%DOTIMESTEP Do a single time step of the system

% Assume forces have been calculated; move the system
% EXPLICIT IN VELOCITY
o.fluid.u = o.fluid.u + (o.dt./o.fluid.m) .* ( ...
    o.F_pre + ...
    o.F_adv + ...
    o.F_vis + ...
    o.F_ext );
o.fluid.v = o.fluid.u + (o.dt./o.fluid.m) .* o.F_bkg;
o.fluid.v = o.fluid.u;

% IMPLICIT IN COORDINATES
o.fluid.r = o.fluid.r + o.dt * o.fluid.v;

% Recalculate things
o.doQuantities;

% Record things
o.T = o.T + o.dt;
o.f = o.f + 1;

end

