function S = lidDrivenCavity(NUM, REY)
%GEN_BOX_PARTICLES Generate coordinates of particles in a box
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

% Adams paper
LEN = 1;
RHO = 1;
VEL = 1;
SND = 10;

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
% Get boundary as opposed to fluid particles
b = any((pts<=0)|(pts>LEN), 2);

% Create fluid particles;
F = Particles(pts(~b,:), RHO, dx.^DIM, KIN);
B = Particles(pts(b, :), RHO, dx.^DIM, KIN);
% Create wrapper
NAME = ['Lid Driven Cavity', ...
    ' Re=', num2str(REY), ...
    ' (', num2str(NUM), 'x', num2str(NUM), ')'];
S = XSPH(NAME, F, B);
% Input data
S.g = [0, -GRA];
S.h = INF*dx;
S.u0 = VEL;
S.c0 = SND * VEL;
S.p0 = -1 * math.tait(0, RHO, 10*S.u0, 1);
S.pb = 2 * S.p0;
S.d0 = RHO;
% Create function calls
S.chooseKernel('quintic');
S.chooseStateEqn('ideal gas');

% Determine time-step
S.dt = .25 * min( [...
    S.h / ( S.c0 + S.u0 ), ...
    ( S.h^2 ) / KIN, ...
    sqrt( S.h / GRA ) ] );


% Set advection(wall) velocity for wall particles
S.bound.v(S.bound.r(:,2)<=0,1) = S.u0;

% Calculate the initial values
S.doQuantities;

end













