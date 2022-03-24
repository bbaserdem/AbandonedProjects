% SPH cavity flow using XSPH
clear variables
addpath functions

% Regularizer
EPS = 1e-10;
% Problem dimension
DIM = 2;

%--------------------------------------------------------%
%  ____                                _                 %
% |  _ \ __ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ ___  %
% | |_) / _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__/ __| %
% |  __/ (_| | | | (_| | | | | | |  __/ ||  __/ |  \__ \ %
% |_|   \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|  |___/ %
%--------------------------------------------------------%

% Length of wall segment.
LEN = 1e-3;
% Particles per edge
NUM = 20;
% Packing of boundary particles
PAC = 2;
% Particle seperation
PDX = LEN/(NUM+(1/PAC));
BDX = PDX/PAC;
% Support length
SUP = 1.25 * PDX;

% Fluid density
RHO = 1e3;
% Kinematic viscocity
VIS = 1e-6;
% Reynolds number
REY = 10;
% Boundary speed
U_0 = REY * VIS / PDX;

% KERNEL: Use wendland kernel
DOM = 2 * SUP;
EQN_KER = @(u) kernel_wendland(u, SUP, DIM, false);
EQN_KERDER = @(u) kernel_wendland(u, SUP, DIM, true );

% BOUNDARY: Lennard-Jones potential
LJ1 = 12;
LJ2 = 4;
LJD = 1e-2;
EQN_BND = @(u) inter_lennard_jones(u, PDX/2, LJ1, LJ2, LJD);

% PRESSURE: Taut equation
GAM = 1;
CSN = 10;
PBK = GAM*((U_0*CSN)^2)/RHO;
EQN_PRE = @(u) inter_taut(u, PBK, RHO, GAM, 0);
P0 = 1 * PBK;

% Time-step
% D_T = .25*min(SUP/(U_0*(1+CSN)), (SUP^2)/(VIS) );
D_T = 2.5e-5;
% Iterations
ITE = 50000;

%----------------------------------------%
%   ___       _ _   _       _ _          %
%  |_ _|_ __ (_) |_(_) __ _| (_)_______  %
%   | || '_ \| | __| |/ _` | | |_  / _ \ %
%   | || | | | | |_| | (_| | | |/ /  __/ %
%  |___|_| |_|_|\__|_|\__,_|_|_/___\___| %
%----------------------------------------%

% Generate particles
[P_POS, B_POS] = gen_box_particles(NUM, LEN, DIM, PAC);

% PARTICLE: Initialize mass, volume and density
P_MVE = zeros(size(P_POS));
P_VIS = VIS * ones(size(P_POS, 1), 1);
P_DEN = RHO * ones(size(P_POS, 1), 1);
P_VOL = (PDX.^DIM) * ones(size(P_POS, 1), 1);
P_MAS = P_VOL .* P_DEN;
P_PRE = EQN_PRE(P_DEN);
P_AVE = P_MVE;

% BOUNDARY: Initialize mass, volume and density
B_MVE = zeros(size(B_POS));
B_VIS = VIS * ones(size(B_POS, 1), 1);
B_MVE(B_POS(:,2)==LEN, 1) = U_0;
B_DEN = RHO * ones(size(B_POS, 1), 1);
B_VOL = (BDX.^DIM) * ones(size(B_POS, 1), 1);
B_MAS = B_VOL .* B_DEN;
B_PRE = EQN_PRE(B_DEN);
B_AVE = B_MAS .* B_MVE;

%------------------------------------------------------%
%   ____  _                 _       _   _              %
%  / ___|(_)_ __ ___  _   _| | __ _| |_(_) ___  _ __   %
%  \___ \| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \  %
%   ___) | | | | | | | |_| | | (_| | |_| | (_) | | | | %
%  |____/|_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_| %
%------------------------------------------------------%
% Utilizing kick-drift-kick scheme

% Initialize
P_MVE_HALF = P_MVE;
P_FOR = zeros(size(P_MVE));
P_P0 = zeros(size(P_MVE));
% Get valid distances
[P_R_IJ, P_HAT_IJ_D, P_I, P_J] = get_dist_pair(P_POS, DOM);
[B_R_IJ, B_HAT_IJ_D, B_I, B_J] = get_dist_bndr(P_POS, B_POS, DOM);

for t = 1:ITE
    % KICK-STEP: Update half-step velocities
    P_MVE_HALF = P_MVE + (.5*D_T./P_MAS).*P_FOR;
    P_AVE = P_MVE_HALF + (.5*D_T./P_MAS).*P_P0;
    
    % DRIFT-STEP: Update particle positions
    P_POS = P_POS + D_T*P_AVE;
    
    % CALCULATION
    % Get valid distances
    [P_R_IJ, P_HAT_IJ_D, P_I, P_J] = get_dist_pair(P_POS, DOM);
    [B_R_IJ, B_HAT_IJ_D, B_I, B_J] = get_dist_bndr(P_POS, B_POS, DOM);
    % Get distance inverses
    P_RINV_IJ = spfun(@(x) 1./x, P_R_IJ);
    B_RINV_IJ = spfun(@(x) 1./x, B_R_IJ);
    % Get kernel and derivative
    P_KER_IJ =    spfun(EQN_KER,    P_R_IJ)+EQN_KER(0)*speye(size(P_POS,1));
    P_KERDER_IJ = spfun(EQN_KERDER, P_R_IJ);
    B_KER_IJ =    spfun(EQN_KER,    B_R_IJ);
    B_KERDER_IJ = spfun(EQN_KERDER, B_R_IJ);
    % Get density, and pressure
    P_DEN = P_MAS.*full(sum(P_KER_IJ, 2)+sum(B_KER_IJ, 2));
    P_PRE = EQN_PRE(P_DEN);
    % Get symmetric viscosity
    P_VIS_IJ = get_avg_har_pair(P_VIS, P_I, P_J);
    B_VIS_IJ = get_avg_har(P_VIS, B_VIS, B_I, B_J);
    % Get symmetric pressure
    P_PRE_IJ = get_avg_weight_pair(P_PRE, P_DEN, P_I, P_J);
    B_PRE_IJ = get_avg_weight(P_PRE, P_DEN, B_PRE, B_DEN, B_I, B_J);
    % Get symmetric volume
    P_VOL_IJ = get_sum_sqr_pair(P_VOL, P_I, P_J);
    B_VOL_IJ = get_sum_sqr(P_VOL, B_VOL, B_I, B_J);
    % Get momentum velocity difference
    P_VEL_IJ_D = get_diff_pair(P_MVE_HALF, P_I, P_J);
    B_VEL_IJ_D = get_diff_uv(P_MVE_HALF, B_MVE, B_I, B_J);
    % Get the drift tensor
    P_A1_IJ_D = get_avg_pair(P_DEN.*P_MVE_HALF, P_I, P_J);
    P_A2_IJ_D = get_avg_pair(P_AVE-P_MVE_HALF, P_I, P_J);
    % Get repulsion force
    B_REP_IJ = spfun(EQN_BND, B_R_IJ);
    % Get background pressure effect
    P_P0_IJ = cell_times_mat(-P0*P_VOL_IJ.*P_KERDER_IJ, P_HAT_IJ_D);
    % Calculate forces on particle
    FORCE.p_pres = cell_contract(cell_times_mat(...
        -1 * P_PRE_IJ .* P_VOL_IJ .* P_KERDER_IJ, P_HAT_IJ_D));
    FORCE.p_drif = cell_contract(cell_times_mat(...
        P_VOL_IJ .* P_KERDER_IJ .* cell_dotprod(P_A2_IJ_D, P_HAT_IJ_D), ...
        P_A1_IJ_D));
    FORCE.p_visc = cell_contract(cell_times_mat(...
        P_VOL_IJ .* P_KERDER_IJ .* P_VIS_IJ .* P_RINV_IJ, P_VEL_IJ_D));
    FORCE.p_back = cell_contract(cell_times_mat(...
        -P0 * P_VOL_IJ .* P_KERDER_IJ, P_HAT_IJ_D));
    % Calculate force
    FORCE.b_pres = cell_contract(cell_times_mat(...
        -1 * B_PRE_IJ .* B_VOL_IJ .* B_KERDER_IJ, B_HAT_IJ_D));
    FORCE.b_visc = cell_contract(cell_times_mat(...
        B_VOL_IJ .* B_KERDER_IJ .* B_VIS_IJ .* B_RINV_IJ, B_VEL_IJ_D));
    FORCE.b_wall = cell_contract(cell_times_mat(...
        B_VOL_IJ .* B_KERDER_IJ .* B_REP_IJ , B_HAT_IJ_D));
    % Actual force calc
    P_FOR_IJ = cell_times_mat(P_VOL_IJ.*P_KERDER_IJ, cell_plus( ...
        cell_times_mat(-1*P_PRE_IJ, P_HAT_IJ_D), ...
        cell_times_mat(cell_dotprod(P_A2_IJ_D, P_HAT_IJ_D), P_A1_IJ_D), ...
        cell_times_mat(P_VIS_IJ.*P_RINV_IJ, P_VEL_IJ_D)));
    P_P0_IJ = cell_times_mat(-P0*P_VOL_IJ.*P_KERDER_IJ, P_HAT_IJ_D);
    B_FOR_IJ = cell_times_mat(B_VOL_IJ.*B_KERDER_IJ, cell_plus( ...
        cell_times_mat(-1*B_PRE_IJ, B_HAT_IJ_D), ...
        cell_times_mat(B_REP_IJ, B_HAT_IJ_D), ...
        cell_times_mat(B_VIS_IJ.*B_RINV_IJ, B_VEL_IJ_D)));
    % Get the non-ij quantities
    P_FOR = (cell_contract(P_FOR_IJ) + cell_contract(B_FOR_IJ));
    P_P0 = cell_contract(P_P0_IJ);
    FORCE.total = P_FOR + P_P0;
    
    % KICK-STEP: Update velocity
    P_MVE = P_MVE_HALF + (.5*D_T./P_MAS).*P_FOR;
end

% Get interactions
[~, ~, P_I, P_J] = get_dist_pair(P_POS, DOM);
[~, ~, B_I, B_J] = get_dist_bndr(P_POS, B_POS, DOM);
% Draw

plot_2D_velocity(P_POS, B_POS, P_MVE, [P_I, P_J], [B_I, B_J]);
plot_2D_forces(P_POS, B_POS, FORCE );







































