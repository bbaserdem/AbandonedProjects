% Bogacz structure

clear variables
load data/mnist.mat

% Functions
tanh_deriv = @(u) 1 - tanh(u).^2;
sigmoid = @(u) 1 ./ ( 1 + exp(-u) );
sigmoid_deriv = @(u) sigmoid(u) .* ( 1 - sigmoid(u) );
relu = @(u) max(0, u);
relu_deriv = @(u) 1 .* ( u > 0 );
linear = @(u) u;
linear_deriv = @(u) ones( size(u) );


lrn_img = lrn_img - mean( lrn_img(:) );
tst_img = tst_img - mean( tst_img(:) );
tst_ind = (1:10) * tst_lab;

% Sample sizes
N_tst = size(tst_img,2);
N_lrn = size(lrn_img,2);

% Function
func = relu;
derv = linear_deriv;
% Runtime parameters
epoc = 10;
time = 100;
delt = 0.1;
momt = 0.7;
rate = .5 / N_lrn;
wdec = 0.0;

% Entry sizes
N0 = size(lrn_img,1);
N1 = size(lrn_lab,1);

% Can be arbitrary
N_layers = [200 100];
N = length(N_layers);

% Layers will be cells, X has extra input, and both has extra output
X = cell(1, N );
E = cell(1, N + 1 );
S = ones(1, N + 1 );

% initialize weights, as cells
W = cell(1, N + 1 );
delW = W;
W{1} = randn( N_layers(1), N0) / sqrt(N_layers(1)+N0);
for a = 2:N
    W{a} = randn( N_layers(a), N_layers(a-1) ) / ...
        sqrt(N_layers(a)+N_layers(a-1));
end
W{end} = randn( N1, N_layers(N) ) / sqrt(N_layers(N)+N1);
for a = 1:length(W)
    delW{a} = zeros( size( W{a} ) );
end


% Initialize statistics
perf = zeros(1,epoc);

for e = 1:epoc
    
    fprintf('Epoch no: %d\n', e);
    tic;
    
    % Flush X and E
    for a = 1:N
        X{a} = zeros( N_layers(a) ,N_lrn);
        E{a} = zeros( N_layers(a) ,N_lrn);
    end
    E{end} = zeros( N1 ,N_lrn);
    
    %speeding up calculations
    E1_right_input = W{1} * func( lrn_img );
    
    % Learning simulation
    for t = 1:time
        % Determine E's
        E{1} = (X{1} - E1_right_input) / S(1);
        for a = 2:N
            E{a} = ( X{a} - W{a} * func( X{a-1} ) ) / S(a);
        end
        E{end} = ( lrn_lab - W{end} * func( X{end} ) ) / S(end);
        % Determine X's
        for a = 1:N
            X{a} = X{a} + delt * ( ...
                ((W{a+1}') * E{a+1}) .* derv(X{a}) - E{a} );
        end
    end
    
    % Weight updates
    delW{1} = momt * delW{1} + ( E{1} * func(lrn_img') );
    for a = 2:length(delW)
        delW{a} = momt * delW{a} + ( E{a} * func(X{a-1}') );
    end
    for a = 1:length(W)
        W{a} = (1-wdec) * W{a} + rate * delW{a};
    end
    
    % Run verification
    % Flush X and E
    for a = 1:N
        X{a} = zeros( N_layers(a) ,N_tst);
        E{a} = zeros( N_layers(a) ,N_tst);
    end
    E{end} = zeros( N1 ,N_tst);
    X_pred = zeros( N1, N_tst );
    
    E1_right_input = W{1} * func( tst_img );
    
    % Simulation
    for t = 1:time
        % Determine E's
        E{1} = ( X{1} - E1_right_input ) / S(1);
        for a = 2:N
            E{a} = ( X{a} - W{a} * func( X{a-1} ) ) / S(a);
        end
        E{end} = ( X_pred - W{end} * func( X{end} ) ) / S(end);
        % Determine X's
        for a = 1:N
            X{a} = X{a} + delt * ( ...
                ((W{a+1}') * E{a+1}) .* derv(X{a}) - E{a} );
        end
        X_pred = X_pred - delt * E{end};
    end
    % get prediction
    [~,X_pred_ind] = max( X_pred );
    perf(e) = 1 - ( sum( X_pred_ind == tst_ind ) / length(tst_ind) );
    fprintf("Error percentage: %.2f%%\n", 100*perf(e));
    toc;
end