function showForces(o, K)
%SHOWFORCES Show velocities on the figure

% Set figure
o.figFor = aux.focusFigure(o.figFor);

% Default to shoving all the forces
if ~exist('K','var')
    K = 'all';
end

clf('reset');
% Draw particles
o.drawParticles;
% Draw forces
o.drawForces(K);
title(['Forces on fluid particles: ', o.name]);

end

