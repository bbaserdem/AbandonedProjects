function showInteractions(o)
%SHOWINTERACTIONS Show particles and connect ones that are interacting

% Set figure
o.figInt = aux.focusFigure(o.figInt);

% Draw particles
o.drawParticles;
% Draw interactions
o.drawIntBorder;
o.drawIntFluid;
title(['Interactions: ', o.name]);

end

