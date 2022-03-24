function calculateMovement(o)
%CALCULATEMOVEMENT Moves the particles (This is the time scheme)

% Total force
f = o.prt_for_pre + o.prt_for_adv + o.prt_for_vis + o.prt_for_ext;

% Update momentum velocity of particles
o.prt_mom = o.prt_mom + (o.sim_step ./ o.prt_mas) .* f;

% Calculate background corrected velocity
o.prt_vel = o.prt_mom + (o.sim_step ./ o.prt_mas) .* o.prt_for_cor;

% Move coordinates
o.prt_pos = o.prt_pos + o.sim_step * o.prt_vel;

% Move one time step
o.sim_iter = o.sim_iter + 1;

end

