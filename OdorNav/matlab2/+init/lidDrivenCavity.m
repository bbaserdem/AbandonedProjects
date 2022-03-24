function S = lidDrivenCavity(NUM, REY)
% Generate coordinates of particles in a box
%#ok<*AGROW>

if ~exist('NUM','var')
    NUM = 50;
end
if ~exist('REY','var')
    REY = 100;
end

% Sane defaults
DIM = 2;
INF = 1;
GRA = 0;
SMO = 1.2;

% Adams paper as defaults
LEN = 1;
RHO = 1;
VEL = 1;
SND = 10 * VEL;

% Calculate kinematic viscosity;
KIN = VEL * LEN / REY;

% Particle seperation
DX = LEN/NUM;

% Initialize object
S = SPH_TV('Lid-driven Cavity');

%-----PARAMETERS
% Referance values
S.par_sep = DX;
S.par_snd = 10 * VEL;
S.par_spd = VEL;
S.par_den = RHO;
S.par_vis = KIN;
S.par_pre = -1 * math.tait(0, S.par_den, S.par_snd, 1);
S.par_vol = DX^DIM;
S.par_mas = S.par_den * S.par_vol;
S.par_grv = GRA * [0, -1];

%------PARTICLES
% Add this many extra particles as border particles
ad = ceil(3*INF);
% Grid seed
seed = ((-DX*(ad-1)):DX:(LEN+DX*ad))';
% Generate all points
pts = zeros(1, 0);
for d = 1:DIM
    pts = [repmat(pts, length(seed), 1), repelem(seed, size(pts,1), 1)];
end
% Identify border points
sto = ~any((pts<=0)|(pts>LEN), 2);
% Insert points
S.insertParticles(pts(sto,:), 'fluid');
S.insertParticles(pts(~sto,:), 'boundary');
% Set wall velocity!
sto = S.prt_pos(:,2) <= 0;
S.prt_vel(sto,1) = VEL;


%-----GEOMETRY
S.geo_min = min(pts, [], 1);
S.geo_max = max(pts, [], 1);
S.geo_inl_min = LEN * ones(1, DIM);
S.geo_inl_max = LEN *zeros(1, DIM);
S.geo_smt = SMO * DX;
S.chooseKernel('quintic');

%-----SIMULATION
S.chooseStateEqn('ideal gas');
S.sim_step = .25 * min( [...
    S.geo_smt / ( S.par_spd + S.par_snd ), ...
    ( S.geo_smt^2 ) / S.par_vis, ...
    sqrt( S.geo_smt / GRA ) ...
    ] );

end













