% SPH cavity flow using XSPH
clear variables

% Regularizer
EPS = 1e-10;


%----------------------------------------%
%   ___       _ _   _       _ _          %
%  |_ _|_ __ (_) |_(_) __ _| (_)_______  %
%   | || '_ \| | __| |/ _` | | |_  / _ \ %
%   | || | | | | |_| | (_| | | |/ /  __/ %
%  |___|_| |_|_|\__|_|\__,_|_|_/___\___| %
%----------------------------------------%

G = 12000;
REST_DENS = 1000;
GAS_CONST = 2000;
H = 16;
MASS = 65;
VISC = 250;
DT = .0008;
T = 100;

% Generate particles
P = InitSPH(H, 20, [0, 1000; 0, 1000]);
P.mass = MASS;
P.grav = P.grav * G;
P.gasConst = GAS_CONST;
P.rhoRefer = REST_DENS;
P.visc = VISC;

% Show particles
P.RenderFrame;




%------------------------------------------------------%
%   ____  _                 _       _   _              %
%  / ___|(_)_ __ ___  _   _| | __ _| |_(_) ___  _ __   %
%  \___ \| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \  %
%   ___) | | | | | | | |_| | | (_| | |_| | (_) | | | | %
%  |____/|_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_| %
%------------------------------------------------------%
% Utilizing kick-drift-kick scheme

for t = 1:1
    P.ComputeDistance;
    P.ComputeDensity;
    P.ComputePressure;
    P.ComputeForces;
    P.Integrate(DT);
    P.RenderFrame;
end







































