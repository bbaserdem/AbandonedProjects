function showParticles(o)
%SHOWPARTICLES Show particles on the figure

% Set figure
o.fig_prt = aux.focusFigure(o.fig_prt);

% Draw particles
o.plotParticles;
title(['System particles: ', o.sim_name, ': Time ', num2str(o.sim_time)]);

end