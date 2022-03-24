function showForces(o, want)
%SHOWFORCES Show particles on the figure

if ~exist('want','var')
    want = 'all';
end

% Set figure
o.fig_for = SPH.getFocusedFig(o.fig_for);

% Draw particles
o.plotParticles;

% Particles
r = o.prt_sim_id;

% Vector holder
v = [];

% Label holder
l = {};

if strcmp(want, 'pre') || strcmp(want, 'all')
    v = [v, o.plotVectors(r, o.prt_for_pre(r,:), [1.00 1.00 0.00])];
    l = {l{:}; 'Pressure'};
end
if strcmp(want, 'vis') || strcmp(want, 'all')
    v = [v, o.plotVectors(r, o.prt_for_vis(r,:), [0.00 1.00 1.00])];
    l = {l{:}; 'Viscous'};
end
if strcmp(want, 'adv') || strcmp(want, 'all')
    v = [v, o.plotVectors(r, o.prt_for_adv(r,:), [1.00 0.00 1.00])];
    l = {l{:}; 'Advective'};
end
if strcmp(want, 'ext') || strcmp(want, 'all')
    v = [v, o.plotVectors(r, o.prt_for_ext(r,:), [0.25 1.00 0.25])];
    l = {l{:}; 'External'};
end
if strcmp(want, 'cor') || strcmp(want, 'all')
    v = [v, o.plotVectors(r, o.prt_for_cor(r,:), [1.00 0.25 0.25])];
    l = {l{:}; 'Correction'};
end
l = l';

title(['Forces (', want, '): ', o.sim_name, ', Time: ', num2str(o.sim_time)]);
legend(v,l);

end