function calculateBoundary(o)
%CALCULATEBOUNDARY Calculates value of things at the boundary particles

% Calculate correction vector for truncation
COR = 1 ./ ( 1e-15 + full(sum(o.Bw_ij,1)') );

% Calculate the (correction) velocity
o.bound.u = (2*o.bound.v) - full((o.Bw_ij')*o.fluid.v).*COR;
% Calculate pressure of boundary particles
K = SparseTensor((1:o.bNum)',ones(o.bNum,1),o.g-o.bound.a,o.bNum,1)';
R = (o.Br_ij.*o.Be_ij).';
o.bound.p = full(sum(...
    ( (o.fluid.p') + (K*R).*(o.fluid.d') ) .* (o.Bw_ij'), 2) ) .* COR;
% Calculate density of boundary particles
o.bound.d = o.density(o.bound.p);
o.bound.s = o.bound.m ./ o.bound.d;

end

