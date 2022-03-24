clear variables
load data/mnist.mat

lrn_img = lrn_img - mean( lrn_img(:) );
tst_img = tst_img - mean( tst_img(:) );

% Nonlinearity
a = 0.1;
nlin = @(u) min( 10, max( 10 * u, 0 ) );

% Build network of;
% IMG(400)
% (G)
% D0(400)
% (W)
% X1(10)
% (A)
% D1(10)

% Sample sizes
N_tst = size(tst_img,2);
N_lrn = size(lrn_img,2);

% Operate under 2 nodes, incoming IMG transformed and identity (G=id)
N0 = size(lrn_img,1);
N1 = size(lrn_lab,1);

% Can be arbitrary later, but it should be good to keep this like it is.
N = N0;

% Keep track of sizes like this
N = [N0, N, N1];

% initialize weights
G_old = eye( N(1) );
G_new = randn( N(2), N(1) ) / ( sqrt( N(2) + N(1) ) / 12 );
W_old = randn( N(3), N(2) ) / ( sqrt( N(3) + N(2) ) / 12 );
W_new = W_old;

% Simulation parameters
E = 10;
T = 100;
dt = 0.05;
tao = Inf;
rate = .005;
decay = 0.05;

% init
performance_old = zeros(1, E);
performance_new = zeros(1, E);
[~,indices] = max(tst_lab);

% Runtime
for epoch = 1:E
    fprintf('Epoch no: %d\n', epoch);
    % Pre-parameters
    D0_old = zeros( N(2), N_lrn );
    X1_old = zeros( N(3), N_lrn );
    D1_old = zeros( N(3), N_lrn );
    D0_new = D0_old;
    X1_new = X1_old;
    D1_new = D1_old;
    
    % Learning, simulate
    for t = 1:T
        % OLD STYLE
        % Set D's
        D0_old = lrn_img - W_old' * nlin( X1_old );
        D1_old = nlin( X1_old )  - lrn_lab;
        % Set X's
        X1_old = X1_old + dt * ( ...
            (-1/tao) * X1_old + ...
            W_old * D0_old - ...
            D1_old );
        % NEW STYLE
        % Set D's
        D0_new = G_new * lrn_img - W_new' * nlin( X1_new );
        D1_new = nlin( X1_new ) - lrn_lab;
        % Set X's
        X1_new = X1_new + dt * ( ...
            (-1/tao) * X1_new + ...
            W_new * D0_new - ...
            D1_new );
    end
    % Teach
    % Old style
    W_old = (1 - decay) * W_old + rate * ( nlin(X1_old) * D0_old' );
    % New style
    G_new = (1 - decay) * G_new + rate * ( D0_new * lrn_img');
    W_new = (1 - decay) * W_new + rate * ( nlin(X1_new) * D0_new' );
    
    % Pre-parameters
    D0_old = zeros( N(2), N_tst );
    X1_old = zeros( N(3), N_tst );
    D1_old = zeros( N(3), N_tst );
    D0_new = D0_old;
    X1_new = X1_old;
    D1_new = D1_old;
    
    % Teaching, simulate
    for t = 1:T
        % OLD STYLE
        % Set D's
        D0_old = tst_img - W_old' * nlin( X1_old );
        D1_old = nlin( X1_old );
        % Set X's
        X1_old = X1_old + dt * ( ...
            (-1/tao) * X1_old + ...
            W_old * D0_old - ...
            D1_old );
        % NEW STYLE
        % Set D's
        D0_new = G_new * tst_img - W_new' * nlin( X1_new );
        D1_new = nlin( X1_new );
        % Set X's
        X1_new = X1_new + dt * ( ...
            (-1/tao) * X1_new + ...
            W_new * D0_new - ...
            D1_new );
    end
    % Teaching determine
    [~,ind_old] = max( D1_old );
    performance_old(epoch) = sum(indices == ind_old) / N_tst;
    [~,ind_new] = max( D1_new );
    performance_new(epoch) = sum(indices == ind_new) / N_tst;
end