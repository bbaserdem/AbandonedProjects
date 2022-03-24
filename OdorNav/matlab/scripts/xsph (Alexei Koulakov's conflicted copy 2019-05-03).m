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
NUM = 5;
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

% Time-step
D_T = 5e-5;
% Iterations
ITE = 100;

% KERNEL: Use wendland kernel
DOM = 2 * SUP;
EQN_KER = @(u) kernel_wendland(u, SUP, DIM, false);
EQN_K_D = @(u) kernel_wendland(u, SUP, DIM, true );

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
P_MAS = P_VOL .* DEN;
P_PRE = EQN_PRE(P_DEN);
P_AVE = P_MVE;

% BOUNDARY: Initialize mass, volume and density
B_MVE = zeros(size(B_POS));
B_VIS = VIS * ones(size(B_POS, 1), 1);
B_MVE(B_POS(:,2)==LEN, 1) = U_0;
B_DEN = RHO * ones(size(B_POS, 1), 1);
B_VOL = (BDX.^DIM) * ones(size(B_POS, 1), 1);
B_MAS = B_VOL .* DEN;
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

% Write functions
cellprod = @(x,y) cellfun( @(a,b) a.*b, x, y, 'UniformOutput', false);
cmprod = @(x,y) cell2mat(cellfun(@(a) a.*x, y, 'UniformOutput', false));
sumcell = @(x) plus([x{:}]);
dotprod = @(x,y) sumcell(cellprod(u,x));

for t = 1:ITE
    
    %-----FORCE CALCULATION-----% BEGIN
    % Get valid distances
    [P_R_IJ, P_R_UV, P_I, P_J] = get_dist_pair(P_POS, DOM);
    [B_R_IJ, B_R_UV, B_I, B_J] = get_dist_bndr(P_POS, P_BND, DOM);
    % Get momentum velocity difference
    P_V_IJ = get_diff_pair(P_MVE, P_I, P_J);
    P_ADV2_IJ = get_diff_pair(P_AVE-P_MVE, P_I, P_J);
    B_V_IJ = get_diff_uv(P_MVE, B_MVE, B_I, B_J);
    % Get kernel and derivative
    P_KER = spfun(EQN_KER, P_R_IJ) + EQN_KER(0)*speye(size(P_POS,1));
    P_K_D = spfun(EQN_K_D, P_R_IJ);
    B_KER = spfun(EQN_KER, B_R_IJ);
    B_K_D = spfun(EQN_K_D, B_R_IJ);
    % PARTICLE: Calculate force
    sto = (P_VOL(P_I).^2) + (P_VOL(P_J).^2);
    P_VOL_IJ = sparse(P_I, P_J, sto);
    sto = ((P_PRE(P_I).*P_DEN(P_J))+(P_PRE(P_I).*P_DEN(P_J)))...
        ./(P_DEN(P_I)+P_DEN(P_J));
    P_PRE_IJ = sparse(P_I, P_J, sto);
    sto = 2*P_VIS(P_I).*P_VIS(P_J)./(P_VIS(P_I)+P_VIS(P_J));
    P_VIS_IJ = sparse(P_I, P_J, sto);
    P_AVD_IJ = .5dotprod
    P_FORCE = cmprod( P_VOL_IJ, ...
        cmprod( -1*P_PRE_IJ
    
end

figure(1);
scatter( COR(:,1), COR(:,2), 150, 'filled' );
hold on;
scatter( BCOR(:,1), BCOR(:,2), 150, 'filled' );
hold off
axis image

% Draw active particle to particle things
if true
    [a,b,d] = find( d_ij );
    l = sub2ind( size(d_ij), a, b );
    dx = full(ex_ij(l));
    dy = full(ey_ij(l));
    x = ((COR(b,2)').*ones(2,1)) + (((d.*dx)').*[0;1]);
    y = ((COR(b,1)').*ones(2,1)) + (((d.*dy)').*[0;1]);
    hold('on');
    plot( x, y, ':k' );
    hold('off');
    [a,b,d] = find( dw_ij );
    l = sub2ind( size(dw_ij), a, b );
    dx = full(ewx_ij(l));
    dy = full(ewy_ij(l));
    x = ((BCOR(b,2)').*ones(2,1)) + (((d.*dx)').*[0;1]);
    y = ((BCOR(b,1)').*ones(2,1)) + (((d.*dy)').*[0;1]);
    hold('on');
    plot( x, y, ':w' );
    hold('off');
end

figure(2);
subplot(1,3,1);
imagesc(DEL_dist); colorbar;
title('Distance calculation error');
subplot(1,3,2);
imagesc(DEL_xvel); colorbar;
title('X velocity calculation error');
subplot(1,3,3);
imagesc(DEL_yvel); colorbar;
title('Y velocity calculation error');
set( gca, 'Color', .5*ones(1,3) );







































