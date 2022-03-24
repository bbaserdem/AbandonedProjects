function showMomenta(o)
%SHOWMOMENTA Show velocities on the figure

% Set figure
o.figMom = aux.focusFigure(o.figMom);

% Draw particles
o.drawParticles;
% Draw velocity arrows on top of the particles; with normalized velocities
fv = o.fluid.u ./ (1e-10+sqrt(sum(o.fluid.u.^2,2)));
bv = o.bound.u ./ (1e-10+sqrt(sum(o.bound.u.^2,2)));
o.drawVectors(fv, bv, false);
% Draw title
title(['Particle momenta direction: ', o.name]);

end

