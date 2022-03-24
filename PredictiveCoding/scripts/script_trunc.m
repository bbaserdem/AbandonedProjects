% Network run

% Start a new network
nettrunk = struct;

% Load MNIST data if not loaded
if ~exist('data_trunk','var')
    data_trunk = load('MNIST_truncated.mat');
    inp_size = size( data_trunk.mnist_img_10, 1);
    out_size = size( data_trunk.mnist_lab_10, 1);
end

% Load data set
nettrunk.img_lrn = data_trunk.mnist_img_10;
nettrunk.lab_lrn = data_trunk.mnist_lab_10;
nettrunk.img_tst = data_trunk.mnist_img_10;
nettrunk.dig_tst = (1:10) * data_trunk.mnist_lab_10;

nettrunk.lrn     = size( nettrunk.img_lrn, 2);

nettrunk.tst     = size( nettrunk.img_tst, 2);
nettrunk.init_k  = 1;

% Network size
nettrunk.Xsize = [10, out_size];
nettrunk.Dsize = [inp_size, nettrunk.Xsize];

% Nonlinearities to use
slin = @(u, k) k*u; % Using scaled linear
relu = @(u, k) max(0, k*u);  % Using ReLU
resu = @(u, k) min(k, max(0, k*u));  % Using Re(ctified)S(witching)U

% Define parameters and dynamics
nettrunk.fix_end = false;
nettrunk.tao = Inf;
nettrunk.momentum = 0.4;
nettrunk.b_learn = true;
nettrunk.alpha = 1;
nettrunk.epoch = 100;
nettrunk.learn = 0.5;
nettrunk.decay = 0.0;
nettrunk.l_dec = .0;
nettrunk.sim_time = 2.0;
nettrunk.sim_step = 0.005;
nettrunk.nlin = relu;

% Run simulation
nettrunk = run_init_weights( nettrunk );
nettrunk = run_train( nettrunk );

% Plot simulation
run_plot( nettrunk );
