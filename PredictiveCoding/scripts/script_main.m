% Network run; wrapper script to run function

% Clear original network
net = struct;
flag_check = false;

% Load MNIST data if not loaded
if ~exist('data','var')
    data = load('MNIST.mat');
    inp_size = size( data.t10k_images, 1);
    out_size = size( data.t10k_oneshot, 1 );
end

% Nonlinearities to used
slin = @(u, k) u*k;         % (S)caled (Lin)ear
relu = @(u, k) max(0, k*u);  % Using ReLU
resu = @(u, k) min(k, max(0, k*u));  % Using Re(ctified)S(witching)U

% Full network inputs
% Img: Image pixels. Lab: Label pixels (onehot vector). Dig: Digits
net.img_lrn = data.train_images;
net.dig_lrn = data.train_labels;
net.lab_lrn = data.train_oneshot;
net.img_tst = data.t10k_images;
net.dig_tst = data.t10k_labels;
net.lab_tst = data.train_oneshot;

% generate

% Define parameters and dynamics
net.epoch = 20;
net.lrn     = 60000;    % size( net.img_lrn, 2); % == 60000
net.tst     = 10000;     % size( net.img_tst, 2); % == 10000
net.Xsize = [out_size];
net.Dsize = [inp_size, net.Xsize];
net.init_k  = .1;
net.fix_end = true;
net.tao = Inf;
net.mom = [0, 0, 0]; % Begin from (1) and linearly raise to (2) until (3)
net.b_learn = false;
net.alpha = 1;
net.learn = 0.001;
net.decay = .0;
net.l_dec = .0;
net.sim_time = 2;
net.sim_step = 0.005;
net.nlin = relu;
net = run_init_weights( net );

if flag_check
    % Copy parameters to truncated network
    nettrunk = net;
    % Truncated network inputs
    nettrunk.img_lrn = data.trunc1_images;
    nettrunk.lab_lrn = data.trunc1_oneshot;
    nettrunk.img_tst = data.trunc1_images;
    nettrunk.dig_tst = data.trunc1_labels;
    % Truncated network specific data
    nettrunk.epoch = 100;
    nettrunk.lrn     = size( nettrunk.img_lrn, 2);
    nettrunk.tst     = size( nettrunk.img_tst, 2);
    % Run truncated network first and plot it
    nettrunk = run_train( nettrunk );
    run_plot( nettrunk );
    input('\nPress return to continue');
end

% Run simulation
net = run_train( net );
run_plot( net );

% END OF SCRIPT














































