% SPH cavity flow
clear variables
range = @(u) [ min(u(:)), max(u(:)) ];

%--------------------------------------------------------%
%  ____                                _                 %
% |  _ \ __ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ ___  %
% | |_) / _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__/ __| %
% |  __/ (_| | | | (_| | | | | | |  __/ ||  __/ |  \__ \ %
% |_|   \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|  |___/ %
%--------------------------------------------------------%

% Problem dimension
DIM = 2;

% Regularizer
EPS = 1e-10;

% Length of wall segment.
L = 1e-3;

% Particles per edge
D = 5;

% Reynolds number
RE = 10;

% Time-step
DT = 5e-5;

% Iterations
IT = 3000;

% Fluid density
RHO = 1e3;

% Kinematic viscocity
VIS = 1e-6;

% K: Interpolator power
K = 0;

% Lennard-Jones potential parameters
LJ1 = 12;
LJ2 = 4;
LJD = 1e-2;

% Tait equation parameters
GAM = 7;
TR0 = 1e3;
TB = 1.013e5;
TAUT = @(u) TB * ( ((u/TR0).^GAM) - 1 );

% Kernel function (Wendland in 2D)
KERF = @(u) (7/(4*pi)) * (u<2) .* ( ((1-u/2).^4) .* (1+2*u) );
KERD = @(u) (7/(4*pi)) * (u<2) .* ( ((1-u/2).^3) .* (-5*u) );
KDOM = 2;

%-----CALCULATED-----%

% Particle seperation
DX = L / D;

% Number of particles
N = D^2;

% Support length
H = 1.25 * DX;

% Boundary speed
U0 = RE * VIS / DX;

% Lennard-Jones potential
LJX = DX/2;
LJFN = @(u) (u<LJX) .* (LJD./u) .* ( ((LJX./u).^LJ1) - ((LJX./u).^LJ2) );

%---------------------------------------------------%
%   ____                        _                   %
%  | __ )  ___  _   _ _ __   __| | __ _ _ __ _   _  %
%  |  _ \ / _ \| | | | '_ \ / _` |/ _` | '__| | | | %
%  | |_) | (_) | |_| | | | | (_| | (_| | |  | |_| | %
%  |____/ \___/ \__,_|_| |_|\__,_|\__,_|_|   \__, | %
%                                            |___/  %
%---------------------------------------------------%
                                           
% Generate boundary particles
sto = L * ( (1:(2*D))' ) / (2*D);
BCOR = [...
    sto, zeros(2*D,1); ...
    L*ones(2*D,1), sto; ...
    L - sto, L*ones(2*D,1);...
    zeros(2*D,1), L - sto ...
    ];
BN = size(BCOR,1);

% Initialize speed at boundary; only top (y==L) has x-velocity U
BVEL = zeros( BN, 2 );
BVEL( BCOR(:,1)==L, 2 ) = U0;

% Initialize mass, volume and density
BDEN = RHO * ones(BN,1);
BVOL = (DX.^DIM) * ones(BN,1);
BMAS = BVOL ./ BDEN;
BPRE = TAUT( BDEN );

%----------------------------------------%
%   ___       _ _   _       _ _          %
%  |_ _|_ __ (_) |_(_) __ _| (_)_______  %
%   | || '_ \| | __| |/ _` | | |_  / _ \ %
%   | || | | | | |_| | (_| | | |/ /  __/ %
%  |___|_| |_|_|\__|_|\__,_|_|_/___\___| %
%----------------------------------------%

% Create ij (j<i) list
IJ = zeros( N*(N-1)/2, 2 );
for i = 1:N
    IJ( ((i-2)*(i-1)/2)+(1:(i-1)), : ) = [ i*ones(i-1,1) , ((1:(i-1))') ];
end
% Create ij list for boundary particles
WIJ = [ kron( ((1:N)') , ones(BN,1) ) , kron( ones(N,1) , ((1:BN)') ) ];

% Insert particles, (particle,Y/X cor)
sto = L * ( (1:D)' - .5 ) / D;
COR = [reshape(ones(D).*sto,[],1), reshape(ones(D).*(sto'),[],1) ];
VEL = zeros( size(COR) );

% Initialize mass, volume and density
DEN = RHO * ones(N,1);
VOL = (DX.^2) * ones(N,1);
MAS = VOL .* DEN;
PRE = TAUT( VOL );

%------------------------------------------------------%
%   ____  _                 _       _   _              %
%  / ___|(_)_ __ ___  _   _| | __ _| |_(_) ___  _ __   %
%  \___ \| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \  %
%   ___) | | | | | | | |_| | | (_| | |_| | (_) | | | | %
%  |____/|_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_| %
%------------------------------------------------------%

% Initialize distance stuff
dist_ij = zeros( size(IJ,1), 1 );
diff_ij = zeros( size(IJ,1), DIM );
udel_ij = zeros( size(IJ,1), DIM );
ehat = cell(1,DIM);
udel = cell(1,DIM,1);
% Initialize distance stuff for wall-particle
Wdist_ij = zeros( size(IJ,1), 1 );
Wdiff_ij = zeros( size(IJ,1), DIM );
Wudel_ij = zeros( size(IJ,1), DIM );
Wehat = cell(1,DIM);
Wudel = cell(1,DIM);

% Write functions
cellprod = @(x,y) cellfun( @(a,b) a.*b, x, y, 'UniformOutput', false);
cmprod = @(x,y) cell2mat(cellfun(@(a) a.*y, x, 'UniformOutput', false));
sumcell = @(x) plus([x{:}]);
dotprod = @(x,y) sumcell(cellprod(u,x));

for t = 1:3
    
    %-----PAIRWISE DISTANCE CALCULATION-----% BEGIN
    % PARTICLE - PARTICLE
    % Get the delta vectors
    sto = true(size(IJ,1),1);
    for d = 1:DIM
        diff_ij(sto,d) = COR(IJ(sto,1),d) - COR(IJ(sto,2),d);
        sto = sto & ((KDOM*H)>abs(diff_ij(:,d)));
    end
    % Calculate distances between points that are not too far
    dist_ij(sto) = sum(diff_ij(sto,:).^2);
    sto(sto) = dist_ij(sto) < ((KDOM*H).^2);
    dist_ij(sto) = sqrt(dist_ij(sto));
    % Put distances in sparse array
    dist = sparse( IJ(sto,1), IJ(sto,2), dist_ij(sto), N, N );
    dist = dist + (dist');
    % Calculate kernel and derivative
    wker = ( spfun( KERF, dist/H ) + (KERF(0)*speye(N)) ) / (H^2);
    gker = spfun( KERD, dist/H ) / (H^3);
    % Calculate hat vector components, and velocities
    udel_ij(sto,:) = VEL(IJ(sto,1),:) - VEL(IJ(sto,2),:);
    for d = 1:DIM
        ehat{d} = sparse( IJ(sto,1), IJ(sto,2), diff_ij(sto,d)./dist_ij(sto), N, N );
        udel{d} = sparse( IJ(sto,1), IJ(sto,2), udel_ij(sto,d), N, N );
        ehat{d} = ehat{d} - (ehat{d}');
        udel{d} = udel{d} - (udel{d}');
    end
    %-----PAIRWISE DISTANCE CALCULATION-----% END
    
    % Do half distance step of Stormer-Verlet
    COR = COR + .5 * DT * VEL;
    DEN_G = sum( ...
        ( DEN .^ (2*K-1) ) .* ...
        (VOL') .* ...
        ( (DEN .* (DEN')).^K ) .* ...
        gker .* ...
        dotprod(ehat,udel) ...
        , 2 );
    DEN = DEN + .5 * DT * DEN_G;
    VOL = MAS ./ DEN;
    PRE = TAUT( DEN );
    
    %-----PAIRWISE DISTANCE CALCULATION-----% BEGIN
    % PARTICLE - PARTICLE
    % Get the delta vectors
    sto = true(size(IJ,1),1);
    for d = 1:DIM
        diff_ij(sto,d) = COR(IJ(sto,1),d) - COR(IJ(sto,2),d);
        sto = sto & ((KDOM*H)>abs(diff_ij(:,d)));
    end
    % Calculate distances between points that are not too far
    dist_ij(sto) = sum(diff_ij(sto,:).^2);
    sto(sto) = dist_ij(sto) < ((KDOM*H).^2);
    dist_ij(sto) = sqrt(dist_ij(sto));
    % Put distances in sparse array
    dist = sparse( IJ(sto,1), IJ(sto,2), dist_ij(sto), N, N );
    dist = dist + (dist');
    % Calculate kernel and derivative
    wker = ( spfun( KERF, dist/H ) + (KERF(0)*speye(N)) ) / (H^2);
    gker = spfun( KERD, dist/H ) / (H^3);
    % Calculate hat vector components, and velocities
    udel_ij(sto,:) = VEL(IJ(sto,1),:) - VEL(IJ(sto,2),:);
    for d = 1:DIM
        ehat{d} = sparse( IJ(sto,1), IJ(sto,2), diff_ij(sto,d)./dist_ij(sto), N, N );
        udel{d} = sparse( IJ(sto,1), IJ(sto,2), udel_ij(sto,d), N, N );
        ehat{d} = ehat{d} - (ehat{d}');
        udel{d} = udel{d} - (udel{d}');
    end
    % PARTICLE - WALL
    % Get the delta vectors
    sto = true(size(WIJ,1),1);
    for d = 1:DIM
        Wdiff_ij(sto,d) = COR(WIJ(sto,1),d) - BCOR(WIJ(sto,2),d);
        sto = sto & ((KDOM*H)>abs(Wdiff_ij(:,d)));
    end
    % Calculate distances between points that are not too far
    Wdist_ij(sto) = sum(Wdiff_ij(sto,:).^2);
    sto(sto) = Wdist_ij(sto) < ((KDOM*H).^2);
    Wdist_ij(sto) = sqrt(Wdist_ij(sto));
    % Put distances in sparse array
    Wdist = sparse( WIJ(sto,1), WIJ(sto,2), Wdist_ij(sto), N, N );
    Wdist = Wdist + (Wdist');
    % Calculate kernel and derivative
    Wwker = ( spfun( KERF, Wdist/H ) ) / (H^2);
    Wgker = spfun( KERD, Wdist/H ) / (H^3);
    % Calculate hat vector components, and velocities
    Wudel_ij(sto,:) = VEL(WIJ(sto,1),:) - BVEL(WIJ(sto,2),:);
    for d = 1:DIM
        Wehat{d} = sparse( WIJ(sto,1), WIJ(sto,2), Wdiff_ij(sto,d)./Wdist_ij(sto), N, N );
        Wudel{d} = sparse( WIJ(sto,1), WIJ(sto,2), Wudel_ij(sto,d), N, N );
        Wehat{d} = Wehat{d} - (Wehat{d}');
        Wudel{d} = Wudel{d} - (Wudel{d}');
    end
    %-----PAIRWISE DISTANCE CALCULATION-----% END
    
    %-----F_EXT (WALL FORCE)-----% BEGIN
    sto = spfun(LJFN, Wwker);
    Fext = cmprod(Wehat, sto);
    %-----F_EXT (WALL FORCE)-----% END
    
    %-----F_DIS (DISSIPATION FORCE)-----% BEGIN
    % PARTICLE - PARTICLE
    sto = 2 * ( 2 + 2 ) * VIS * ...
        ( VOL .* (VOL') ) .* ...
        dotprod(ehat,udel) .* ...
        gker .* ...
        ( spfun( @(u) 1./(EPS+u), dist ) );
    Fdis = cmprod(ehat, sto);
    % PARTICLE - WALL
    sto = 2 * ( 2 + 2 ) * VIS * ...
        ( VOL .* (BVOL') ) .* ...
        dotprod(Wehat,Wudel) .* ...
        Wgker .* ...
        ( spfun( @(u) 1./(EPS+u), Wdist ) );
    Fdis = Fdis + cmprod(Wehat, sto);
    %-----F_DIS (DISSIPATION FORCE)-----% END
    
    %-----F_INT (INTERNAL FORCES)-----% BEGIN
    % PARTICLE - PARTICLE
    sto = -1 * ...
        ( MAS .* (MAS') ) .* ...
        ( (PRE.*((DEN').^(2*K))) + ((DEN.^(2*K)).*(PRE')) ) .* ...
        gker .* ...
        ( (DEN.*(DEN')).^(-1-K) );
    Fint = cmprod(ehat, sto);
    % PARTICLE - WALL
    sto = -1 * ...
        ( MAS .* (BMAS') ) .* ...
        ( (PRE.*((BDEN').^(2*K))) + ((DEN.^(2*K)).*(BPRE')) ) .* ...
        Wgker .* ...
        ( (DEN.*(BDEN')).^(-1-K) );
    Fint = Fint + cmprod(Wehat, sto);
    %-----F_INT (INTERNAL FORCES)-----% END
    
    %-----VELOCITY CALCULATION-----%
    F = Fext + Fdis + Fint;
    VEL = VEL + DT * F ./ MAS;
    
    %-----DISTANCE CALCULATION-----%
    COR = COR + .5 * DT * VEL;
    DEN_G = sum( ...
        ( DEN .^ (2*K-1) ) .* ...
        (VOL') .* ...
        ( (DEN .* (DEN')).^K ) .* ...
        g_ij .* ...
        ( (ex_ij.*ux_ij) + (ey_ij.*uy_ij) ) ...
        , 2 );
    DEN = DEN + .5 * DT * DEN_G;
    VOL = MAS ./ DEN;
    PRE = TAUT( DEN );
    
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







































