function showParticles(o)
%SHOWPARTICLES Show particles on the figure

% Set figure
o.figPar = aux.focusFigure(o.figPar);

% Draw particles
o.drawParticles;
title(['System particles: ', o.name, ': Time ', num2str(o.T)]);

end

