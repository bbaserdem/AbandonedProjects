function S = lidDrivenCavity2(NUM, REY)
% Generate coordinates of particles in a box, using the second method
%#ok<*AGROW>
S = SPH;

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

% Adams paper
LEN = 1;
RHO = 1;
VEL = 1;
SND = 10 * VEL;

% Calculate kinematic viscosity;
KIN = VEL * LEN / REY;

% Add this many extra particles as border particles
ad = ceil(3*INF);
% Particle seperation
dx = LEN/NUM;
% Grid seed
seed = ((-dx*(ad-1)):dx:(LEN+dx*ad))';
% Generate all points
pts = zeros(1, 0);
for d = 1:DIM
    pts = [repmat(pts, length(seed), 1), repelem(seed, size(pts,1), 1)];
end

% Insert points
S.cor = pts;

% Set boundary flags
S.f_bound = any((pts<=0)|(pts>LEN), 2);
S.f_fluid = ~S.f_bound;
S.f_inlet = false(size(pts));
S.f_outlet= false(size(pts));

% Set title
S.name = ['Lid Driven Cavity', ...
    ' Re=', num2str(REY), ...
    ' (', num2str(NUM), 'x', num2str(NUM), ')'];

% Input data
S.stepNo = 0;
S.time = 0;
S.dimension = 2;
S.gravity = [0 -GRA];
S.sound = SND;
S.corMin = min(S.cor, [], 1);
S.corMax = max(S.cor, [], 1);
% Referance values
S.refSpeed = VEL;
S.refDensity = RHO;
S.refFriction = KIN;
S.refPressure = -1 * math.tait(0, S.refDensity, S.sound, 1);
% Create function calls
S.chooseKernel('quintic');
S.chooseStateEqn('ideal gas');
% Determine time-step
S.stepSize = .25 * min( [...
    S.smoothing / ( S.refSpeed + S.sound ), ...
    ( S.smoothing^2 ) / S.refFriction, ...
    sqrt( S.smoothing / GRA ) ] );

% Initialize variables
S.vel = zeros(size(S.cor));
S.mom = zeros(size(S.cor));
S.den = RHO * ones(size(S.cor));
S.pre = zeros(size(S.cor));
S.mas = (dx^S.dimension)*RHO*ones(size(S.cor));
S.vol = S.mas ./ S.den;
S.acc = zeros(size(S.cor));
S.fri = KIN * ones(size(S.cor));

% Set wall velocity
S.mom(S.cor(:,2)<=0,1) = S.refSpeed;

end













