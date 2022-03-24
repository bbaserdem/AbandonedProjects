% SPH cavity flow
clear variables
range = @(u) [ min(u(:)), max(u(:)) ];

%---------------------------------------------------------%
%   ____                                _                 %
%  |  _ \ __ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ ___  %
%  | |_) / _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__/ __| %
%  |  __/ (_| | | | (_| | | | | | |  __/ ||  __/ |  \__ \ %
%  |_|   \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|  |___/ %
%---------------------------------------------------------%

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
BVOL = (DX.^2) * ones(BN,1);
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

% Initialize distance matrix
dist =   zeros( size(IJ,1), 1 );
uxdiff = zeros( size(IJ,1), 1 );
uydiff = zeros( size(IJ,1), 1 );
distw =   zeros( size(WIJ,1), 1 );
uwxdiff = zeros( size(WIJ,1), 1 );
uwydiff = zeros( size(WIJ,1), 1 );

for t = 1:350
    
    %-----PAIRWISE DISTANCE CALCULATION-----% BEGIN
    % PARTICLE - PARTICLE
    % Get the delta vectors
    xdiff = COR(IJ(:,1),1) - COR(IJ(:,2),1);
    ydiff = COR(IJ(:,1),2) - COR(IJ(:,2),2);
    % Calculate distances between points that are not too far
    sto = (abs(xdiff)<(KDOM*H)) & (abs(ydiff)<(KDOM*H));
    dist(sto) = xdiff(sto).^2 + ydiff(sto).^2;
    % Get index of points that are within range
    sto(sto) = dist(sto) < ((KDOM*H).^2);
    dist(sto) = sqrt(dist(sto));
    % Put distances in sparse array
    d_ij = sparse( IJ(sto,1), IJ(sto,2), dist(sto), N, N );
    d_ij = d_ij + (d_ij');
    % Calculate kernel and derivative
    w_ij = ( spfun( KERF, d_ij/H ) + (KERF(0)*speye(N)) ) / (H^2);
    g_ij = spfun( KERD, d_ij/H ) / (H^3);
    % Calculate hat vector components
    ex_ij = sparse( IJ(sto,1), IJ(sto,2), xdiff(sto)./dist(sto), N, N );
    ex_ij = ex_ij - (ex_ij');
    ey_ij = sparse( IJ(sto,1), IJ(sto,2), ydiff(sto)./dist(sto), N, N );
    ey_ij = ey_ij - (ey_ij');
    % Calculate velocity components
    uxdiff(sto) = VEL(IJ(sto,1),1) - VEL(IJ(sto,2),1);
    uydiff(sto) = VEL(IJ(sto,1),1) - VEL(IJ(sto,2),1);
    ux_ij = sparse( IJ(sto,1), IJ(sto,2), uxdiff(sto), N, N );
    ux_ij = ux_ij - (ux_ij');
    uy_ij = sparse( IJ(sto,1), IJ(sto,2), uydiff(sto), N, N );
    uy_ij = uy_ij - (uy_ij');
    %-----PAIRWISE DISTANCE CALCULATION-----% END
    
    % Do half distance step of Stormer-Verlet
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
    
    %-----PAIRWISE DISTANCE CALCULATION-----% BEGIN
    % PARTICLE - PARTICLE
    % Get the delta vectors
    xdiff = COR(IJ(:,1),2) - COR(IJ(:,2),2);
    ydiff = COR(IJ(:,1),1) - COR(IJ(:,2),1);
    % Calculate distances between points that are not too far
    sto = (abs(xdiff)<(KDOM*H)) & (abs(ydiff)<(KDOM*H));
    dist(sto) = xdiff(sto).^2 + ydiff(sto).^2;
    % Get index of points that are within range
    sto(sto) = dist(sto) < ((KDOM*H).^2);
    dist(sto) = sqrt(dist(sto));
    % Put distances in sparse array
    d_ij = sparse( IJ(sto,1), IJ(sto,2), dist(sto), N, N );
    d_ij = d_ij + (d_ij');
    % Calculate kernel and derivative
    w_ij = ( spfun( KERF, d_ij/H ) + (KERF(0)*speye(N)) ) / (H^2);
    g_ij = spfun( KERD, d_ij/H ) / (H^3);
    % Calculate hat vector components
    ex_ij = sparse( IJ(sto,1), IJ(sto,2), xdiff(sto)./dist(sto), N, N );
    ex_ij = ex_ij - (ex_ij');
    ey_ij = sparse( IJ(sto,1), IJ(sto,2), ydiff(sto)./dist(sto), N, N );
    ey_ij = ey_ij - (ey_ij');
    % Calculate velocity components
    uxdiff(sto) = VEL(IJ(sto,1),2) - VEL(IJ(sto,2),2);
    uydiff(sto) = VEL(IJ(sto,1),1) - VEL(IJ(sto,2),1);
    ux_ij = sparse( IJ(sto,1), IJ(sto,2), uxdiff(sto), N, N );
    ux_ij = ux_ij - (ux_ij');
    uy_ij = sparse( IJ(sto,1), IJ(sto,2), uydiff(sto), N, N );
    uy_ij = uy_ij - (uy_ij');
    % PARTICLE - WALL
    % Calculate force from wall to particle
    xwdiff = COR(WIJ(:,1),2) - BCOR(WIJ(:,2),2);
    ywdiff = COR(WIJ(:,1),1) - BCOR(WIJ(:,2),1);
    sto = (abs(xwdiff)<(KDOM*H)) & (abs(ywdiff)<(KDOM*H));
    distw(sto) = xwdiff(sto).^2 + ywdiff(sto).^2;
    % Get index of points that are within range
    sto(sto) = distw(sto) < ((KDOM*H).^2);
    distw(sto) = sqrt(distw(sto));
    % Put distances in sparse array
    ewx_ij = sparse( WIJ(sto,1), WIJ(sto,2), xwdiff(sto)./distw(sto), N, BN );
    ewy_ij = sparse( WIJ(sto,1), WIJ(sto,2), ywdiff(sto)./distw(sto), N, BN );
    dw_ij = sparse( WIJ(sto,1), WIJ(sto,2), distw(sto), N, BN );
    ww_ij = spfun( KERF, dw_ij/H ) / (H^2);
    gw_ij = spfun( KERD, dw_ij/H ) / (H^3);
    % Calculate velocity
    uwxdiff = VEL(WIJ(sto,1),2) - BVEL(WIJ(sto,2),2);
    uwydiff = VEL(WIJ(sto,1),1) - BVEL(WIJ(sto,2),1);
    uwx_ij = sparse( WIJ(sto,1), WIJ(sto,2), uwxdiff, N, BN );
    uwy_ij = sparse( WIJ(sto,1), WIJ(sto,2), uwydiff, N, BN );
    %-----PAIRWISE DISTANCE CALCULATION-----% END
    
    %-----F_EXT (WALL FORCE)-----% BEGIN
    sto = spfun( LJFN, dw_ij);
    FXext = sum( ewx_ij .* sto , 2 );
    FYext = sum( ewy_ij .* sto , 2 );
    %-----F_EXT (WALL FORCE)-----% END
    
    %-----F_DIS (DISSIPATION FORCE)-----% BEGIN
    % PARTICLE - PARTICLE
    sto = 2 * ( 2 + 2 ) * VIS * ...
        ( VOL .* (VOL') ) .* ...
        ( (ex_ij.*ux_ij) + (ey_ij.*uy_ij) ) .* ...
        g_ij .* ...
        ( spfun( @(u) 1./(EPS+u), d_ij ) );
    FXdis = sum( ex_ij .* sto , 2 );
    FYdis = sum( ey_ij .* sto , 2 );
    % PARTICLE - WALL
    sto = 2 * ( 2 + 2 ) * VIS * ...
        ( VOL .* (BVOL') ) .* ...
        ( (ewx_ij.*uwx_ij) + (ewy_ij.*uwy_ij) ) .* ...
        gw_ij .* ...
        ( spfun( @(u) 1./(EPS+u), dw_ij ) );
    FXdis = FXdis + sum( ewx_ij .* sto , 2 );
    FYdis = FYdis + sum( ewy_ij .* sto , 2 );
    %-----F_DIS (DISSIPATION FORCE)-----% END
    
    %-----F_INT (INTERNAL FORCES)-----% BEGIN
    % PARTICLE - PARTICLE
    sto = -1 * ...
        ( MAS .* (MAS') ) .* ...
        ( (PRE.*((DEN').^(2*K))) + ((DEN.^(2*K)).*(PRE')) ) .* ...
        g_ij .* ...
        ( (DEN.*(DEN')).^(-1-K) );
    FXint = sum( ex_ij .* sto , 2 );
    FYint = sum( ey_ij .* sto , 2 );
    % PARTICLE - WALL
    sto = -1 * ...
        ( MAS .* (BMAS') ) .* ...
        ( (PRE.*((BDEN').^(2*K))) + ((DEN.^(2*K)).*(BPRE')) ) .* ...
        gw_ij .* ...
        ( (DEN.*(BDEN')).^(-1-K) );
    FXint = FXint + sum( ewx_ij .* sto , 2 );
    FYint = FYint + sum( ewy_ij .* sto , 2 );
    %-----F_INT (INTERNAL FORCES)-----% END
    
    %-----VELOCITY CALCULATION-----%
    F = [ FYext + FYdis + FYint , FXext + FXdis + FXint ];
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
    set( gca, 'Color', .5*ones(1,3) );
end







































