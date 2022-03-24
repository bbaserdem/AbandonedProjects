function S = randomPoints(NUM)
% Generate coordinates of particles in a box
%#ok<*AGROW>

D = 2;
if ~exist('NUM','var')
    NUM = 10;
end

% Initialize object
S = SPH_TV('Random points');

%-----PARAMETERS
% Referance values
S.par_snd = 1;
S.par_spd = 1;
S.par_den = 1;
S.par_vis = 1;
S.par_pre = 1;
S.par_mas = 1;
S.par_grv = zeros(1,D);

%------PARTICLES
S.insertParticles(rand(NUM, D), 'fluid');

%-----GEOMETRY
S.geo_min = zeros(1,D);
S.geo_max = ones(1,D);
S.geo_inl_min = ones(1, D);
S.geo_inl_max = zeros(1, D);
S.geo_smt = .1;
S.chooseKernel('quintic');

%-----SIMULATION
S.chooseStateEqn('ideal gas');

end













