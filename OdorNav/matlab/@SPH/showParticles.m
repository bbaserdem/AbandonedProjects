function showParticles(o)
%SHOWPARTICLES Show particles on the figure

% Set figure
o.fig_particles = aux.focusFigure(o.fig_particles);

% Draw particles
o.plotParticles;
title(['System particles: ', o.name, ': Time ', num2str(o.time)]);

end

