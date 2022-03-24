function showVelocities(o, want)
%SHOWVELOCITIES Show velocities on the figure

% Set figure
o.fig_mom = aux.focusFigure(o.fig_mom);

% Draw particles
o.plotParticles;
title(['Velocities(', want, '): ', o.sim_name, ', time: ', num2str(o.sim_time)]);

end