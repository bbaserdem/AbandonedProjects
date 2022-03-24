function calculateForce(o)
%CALCULATEFORCE Calculate all the forces effecting the system

% Initialize
o.F_pre = zeros( o.fNum, o.dim);
o.F_adv = zeros( o.fNum, o.dim);
o.F_vis = zeros( o.fNum, o.dim);
o.F_ext = zeros( o.fNum, o.dim);
o.F_bkg = zeros( o.fNum, o.dim);

% Get prefactor;
F_PRE = (((o.fluid.s).^2)+((o.fluid.s').^2)) .* o.Fg_ij;
B_PRE = (((o.fluid.s).^2)+((o.bound.s').^2)) .* o.Bg_ij;

% PRESSURE
F_SYMPRE = -o.fluid.getSymPressure(o.Fi_ij, o.Fj_ij);
B_SYMPRE = -o.fluid.getSymPressure(o.Bi_ij, o.Bj_ij, o.bound);
o.F_pre = ...
    sum( F_PRE .* F_SYMPRE .* o.Fe_ij ) + ...
    sum( B_PRE .* B_SYMPRE .* o.Be_ij );

% ADVECTIVE DIFFUSION
F_SYMADV = -o.fluid.getSymAdvection(o.Fi_ij, o.Fj_ij);
o.F_adv = sum( F_PRE .* (F_SYMADV * o.Fe_ij) );

% VISCOSITY
F_SYMFRC = -o.fluid.getSymFriction(o.Fi_ij, o.Fj_ij);
B_SYMFRC = -o.fluid.getSymFriction(o.Bi_ij, o.Bj_ij, o.bound);
F_VELDIF = o.fluid.getDiffVelocity(o.Fi_ij, o.Fj_ij) .* ...
    spfun(@(x) x.^(-1), o.Fr_ij);
B_VELDIF = o.fluid.getDiffVelocity(o.Bi_ij, o.Bj_ij, o.bound) .* ...
    spfun(@(x) x.^(-1), o.Br_ij);
o.F_vis = ...
    - sum( F_PRE .* F_SYMFRC .* F_VELDIF ) + ...
    - sum( B_PRE .* B_SYMFRC .* B_VELDIF );

% EXTERNAL
o.F_ext = o.g + o.fluid.f;

% BACKGROUND CORRECTION
o.F_bkg = o.pb * ( sum( F_PRE .* o.Fe_ij ) + sum( B_PRE .* o.Be_ij ) );

end

