function [ n ] = run_train( n )
%RUN_TRAIN Simulation of the full network

% Generate the index to produce progress prompts
every = floor( n.epoch / 20 );
% Generate random indices to record final epoch behaviour
n.rec_lrn = randi( n.lrn );
n.rec_tst = randi( n.tst );

% If tao doesn't exist, initialize to 1
if ~isfield( n, 'tao' )
    n.tao = 1;
end
% If b learning is not turned on, default to no learning
if ~isfield( n, 'b_learn' )
    n.b_learn = false;
end
% If momentum variable doesn't exist, initialize to 0
if ~isfield( n, 'mom' )
    n.mom = [0, 0, 1];
    momentum = zeros( 1, n.epoch);
else
    momentum = n.mom(2) * ones(1, n.epoch);
    momentum(1:n.mom(3)) = n.mom(1) + ((n.mom(2)-n.mom(1))/n.mom(3)) * ...
        0:(n.mom(3)-1);
end

fprintf('-----\n\tRUN DATA\n');
fprintf('\nTotal epochs          : %d', n.epoch    );
fprintf('\nEpoch size            : %d/%d', n.lrn, size( n.img_lrn, 2) );
fprintf('\nNumber of layers      : %d', n.L_no     );
fprintf('\nSteepness             : %d', n.alpha    );
fprintf('\nSimulation runtime    : %d', n.sim_time );
fprintf('\nSimulation step time  : %d', n.sim_step );
fprintf('\nTime constant         : %d', n.tao );
fprintf('\nWeight initial scaler : %d', n.init_k   );
fprintf('\nMomentum              : %d->(%d steps)->%d', n.mom(1), n.mom(3), n.mom(2));
fprintf('\nLearning rate         : %d', n.learn    );
fprintf('\nLearning rate decay   : %d', n.l_dec    );
fprintf('\nWeight decay          : %d', n.decay    );
fprintf('\nRecorded learn ID     : %d', n.rec_lrn  );
fprintf('\nRecorded test ID      : %d', n.rec_tst  );
fprintf('\nA weights are ID?     : ');
if n.fix_end; fprintf('True'); else; fprintf('False'); end
fprintf('\nB learning on?        : ');
if n.b_learn; fprintf('True\n'); else; fprintf('False\n'); end

% Initialize layers
X_lrn = cell( n.L_no,     3 );
X_tst = cell( n.L_no,     3 ); % 1-activation(x), 2-response(f), 3-x dot
X_nAt = cell( n.L_no,     3 );
D_lrn = cell( n.L_no + 1, 1 );
D_tst = cell( n.L_no + 1, 1 );
D_nAt = cell( n.L_no + 1, 1 );
% Initialize gradients
W_grad = cell( size( n.W ) );
if ~n.fix_end
    A_grad = cell( size( n.A ) );
end
if n.b_learn
    B_grad = cell( size( n.B ) );
end
for a = 1 : n.L_no
    W_grad{a} = zeros( size( n.W{a} ) );
    if ~ n.fix_end
        A_grad{a} = zeros( size( n.A{a} ) );
    end
    if n.b_learn
        B_grad{a} = zeros( size( n.B{a} ) );
    end
end

% Begin counting
tic;

% If sidesample is requested, turn flag on
if n.lrn == size( n.img_lrn, 2)
    subsample_lrn = false;
    image_lrn = n.img_lrn;
    label_lrn = n.lab_lrn;
else
    subsample_lrn = true;
end
if n.tst <= size( n.img_tst, 2)
    image_tst = n.img_tst(:,1:n.tst);
    digit_tst = n.dig_tst(1:n.tst);
else
    error('Expect testing size <= testing set size.');
end

% History recording; record the evolution of the last iteration of the
% first input from the sets
n.Xhist_lrn = cell( n.L_no, 1 );
n.Xhist_tst = cell( n.L_no, 2 );
n.Dhist_lrn = cell( n.L_no+1, 1 );
n.Dhist_tst = cell( n.L_no+1, 2 );
for b = 1:n.X_no
    n.Xhist_lrn{b}   = zeros(n.Xsize(b), length( 0 : n.sim_step : n.sim_time ));
    n.Xhist_tst{b,1} = zeros(n.Xsize(b), length( 0 : n.sim_step : n.sim_time ));
    n.Xhist_tst{b,2} = zeros(n.Xsize(b), length( 0 : n.sim_step : n.sim_time ));
end
for b = 1:n.D_no
    n.Dhist_lrn{b}   = zeros(n.Dsize(b), length( 0 : n.sim_step : n.sim_time ));
    n.Dhist_tst{b,1} = zeros(n.Dsize(b), length( 0 : n.sim_step : n.sim_time ));
    n.Dhist_tst{b,2} = zeros(n.Dsize(b), length( 0 : n.sim_step : n.sim_time ));
end

% Resume previous simulation if runtime is halted
if ~isfield( n, 'last_ep' )
    n.last_ep = 1;
    n.errleg ={'Max          ', 'Max, fb. dis.',...
        'Min          ', 'Min, fb. dis.'};
    n.err = ones( n.epoch, 4);
    r = n.learn / n.lrn;
    n.Dav_lrn = zeros( n.epoch, n.D_no );
    n.Xav_lrn = zeros( n.epoch, n.X_no );
    n.Dav_tst = zeros( n.epoch, n.D_no );
    n.Xav_tst = zeros( n.epoch, n.X_no );
    n.Dav_nAt = zeros( n.epoch, n.D_no );
    n.Xav_nAt = zeros( n.epoch, n.X_no );
    % Record weight updates history (1st L2norm, 2nd is ^2 mean, 3rd stdev
    n.Wlen =          cell( n.L_no, 1 );
    [n.Wlen{:}] =     deal( zeros( n.epoch, 3) );
    if ~n.fix_end
        n.Alen =      cell( n.L_no, 1 );
        [n.Alen{:}] = deal( zeros( n.epoch, 3) );
    end
    if n.b_learn
        n.Blen =      cell( n.L_no, 1 );
        [n.Blen{:}] = deal( zeros( n.epoch, 3) );
    end
else
    r = n.learn * ( 1 - n.l_dec )^(n.last_ep - 1);
    fprintf('\nHalted after epoch #%d, resuming . . .', n.last_ep - 1);
end

for a = n.last_ep : n.epoch
    n.last_ep = a;
    if (every <= 1) || ( 1 == mod(a, every) )
        fprintf('\n-----\nEpoch no: %3d\n', a);
    end
    
    % grab subsample of data if required
    if subsample_lrn
        key = randi( size( n.img_lrn, 2), 1, n.lrn);
        image_lrn = n.img_lrn(:,key);
        label_lrn = n.lab_lrn(:,key);
    end
    
    % Clean out variables
    for b = 1:length(n.Xsize)
        X_lrn{b,1} = zeros( n.Xsize(b), n.lrn);
        X_tst{b,1} = zeros( n.Xsize(b), n.tst);
        X_nAt{b,1} = zeros( n.Xsize(b), n.tst);
    end
    for b = 1:length(n.Dsize)
        D_lrn{b} = zeros( n.Dsize(b), n.lrn);
        D_tst{b} = zeros( n.Dsize(b), n.tst);
        D_nAt{b} = zeros( n.Dsize(b), n.tst);
    end
    
    fprintf( 'Training . . .\n' );
    % Simulating all inputs
    for t = 1 : length( 0 : n.sim_step : n.sim_time )
        % Calculate F's
        for b = 1 : n.X_no
            X_lrn{b,2} = n.nlin( X_lrn{b,1} - n.B{b}, n.alpha );
        end
        % Calculate D's
        D_lrn{1} = image_lrn - n.W{1}' * X_lrn{1,2};
        for b = 2 : n.D_no-1
            D_lrn{b} = n.A{b-1} * X_lrn{b-1,2} - n.W{b}' * X_lrn{b,2};
        end
        D_lrn{end} = n.A{end} * X_lrn{end,2} - label_lrn;
        % Calculate X-dot
        for b = 1 : n.X_no
            X_lrn{b,3} = - (1/n.tao) * (X_lrn{b,1} + n.B{b} ) + ...
                n.W{b} * D_lrn{b}  - n.A{b}' * D_lrn{b+1};
        end
        % Update X's
        for b = 1 : n.X_no
            X_lrn{b,1} = X_lrn{b,1} + n.sim_step * X_lrn{b,3};
        end
        
        % Record if final epoch
        if a == n.epoch
            for b = 1 : n.X_no
                n.Xhist_lrn{b}(:,t) = X_lrn{b,1}(:,n.rec_lrn);
            end
            for b = 1 : n.D_no
                n.Dhist_lrn{b}(:,t) = D_lrn{b}(:,n.rec_lrn);
            end
        end
    end
    
    % Apply weight updates
    for b = 1 : n.L_no
        % Calculate W
        W_grad{b} = momentum(a) * W_grad{b} - X_lrn{b,2} * D_lrn{b}';
        n.Wlen{b}(a,1) = norm( W_grad{b}(:) ) * r;
        n.Wlen{b}(a,2) = mean( W_grad{b}(:) ) * r;
        n.Wlen{b}(a,3) =  std( W_grad{b}(:) ) * r;
        n.W{b} = (1 - n.decay) * n.W{b} - r * W_grad{b};
        % Calculate A if not fixed
        if ~n.fix_end
            A_grad{b} = momentum(a) * A_grad{b} + D_lrn{b+1} * X_lrn{b,2}';
            n.Alen{b}(a,1) = norm( A_grad{b}(:) ) * r;
            n.Alen{b}(a,2) = mean( A_grad{b}(:) ) * r;
            n.Alen{b}(a,3) =  std( A_grad{b}(:) ) * r;
            n.A{b} = (1 - n.decay) * n.A{b} - r * A_grad{b};
        end
        % Calculate B if learning
        if n.b_learn
            B_grad{b} = momentum(a) * B_grad{b} - sum( X_lrn{b,3}, 2 );
            n.Blen{b}(a,1) = norm( B_grad{b}(:) ) * r;
            n.Blen{b}(a,2) = mean( B_grad{b}(:) ) * r;
            n.Blen{b}(a,3) =  std( B_grad{b}(:) ) * r;
            n.B{b} = (1 - n.decay) * n.B{b} - r * B_grad{b};
        end
    end
    r = r * ( 1 - n.l_dec );
    
    fprintf( 'Testing . . .\n' );
    
    % Simulating all inputs
    for t = 1 : length( 0 : n.sim_step : n.sim_time )
        % Calculate F's
        for b = 1 : n.X_no
            X_tst{b,2} = n.nlin( X_tst{b,1} - n.B{b}, n.alpha );
            X_nAt{b,2} = n.nlin( X_nAt{b,1} - n.B{b}, n.alpha );
        end
        % Calculate D's
        D_tst{1} = image_tst - n.W{1}' * X_tst{1,2};
        D_nAt{1} = image_tst - n.W{1}' * X_nAt{1,2};
        for b = 2 : n.D_no-1
            D_tst{b} = n.A{b-1} * X_tst{b-1,2} - n.W{b}' * X_tst{b,2};
            D_nAt{b} = n.A{b-1} * X_nAt{b-1,2} - n.W{b}' * X_nAt{b,2};
        end
        D_tst{end} = n.A{end} * X_tst{end,2};
        D_nAt{end} = n.A{end} * X_nAt{end,2};
        % Calculate X-dot
        for b = 1 : n.X_no
            X_tst{b,3} = - (1/n.tao) * (X_tst{b,1} + n.B{b}) + ...
                n.W{b} * D_tst{b} - n.A{b}' * D_tst{b+1};
            if b == n.L_no; break; end
            X_nAt{b,3} = - (1/n.tao) * (X_nAt{b,1} + n.B{b} ) + ...
                n.W{b} * D_nAt{b} - n.A{b}' * D_nAt{b+1};
        end
        X_nAt{end,3} = - (1/n.tao) * (X_nAt{end,1} + n.B{end} ) + ...
            n.W{end} * D_nAt{end-1};
        % Update X's
        for b = 1 : n.X_no
            X_tst{b,1} = X_tst{b,1} + n.sim_step * X_tst{b,3};
            X_nAt{b,1} = X_nAt{b,1} + n.sim_step * X_nAt{b,3};
        end
        % Record testing session
        for b = 1 : n.X_no
            n.Xhist_tst{b,1}(:,t) = X_tst{b,1}(:,n.rec_tst);
            n.Xhist_tst{b,2}(:,t) = X_nAt{b,1}(:,n.rec_tst);
            n.Dhist_tst{b,1}(:,t) = D_tst{b}(:,n.rec_tst);
            n.Dhist_tst{b,2}(:,t) = D_nAt{b}(:,n.rec_tst);
        end
        for b = 1 : n.D_no
            n.Dhist_tst{b,1}(:,t) = D_tst{b}(:,n.rec_tst);
            n.Dhist_tst{b,2}(:,t) = D_nAt{b}(:,n.rec_tst);
        end
    end
    
    % Get global layer averages for the full epoch
    for b = 1 : n.X_no
        n.Xav_lrn(a, b) = mean( X_lrn{b,1}(:) .^ 2 );
        n.Xav_tst(a, b) = mean( X_tst{b,1}(:) .^ 2 );
        n.Xav_nAt(a, b) = mean( X_nAt{b,1}(:) .^ 2 );
    end
    for b = 1 : n.D_no
        n.Dav_lrn(a, b) = mean( D_lrn{b}(:) .^ 2 );
        n.Dav_tst(a, b) = mean( D_tst{b}(:) .^ 2 );
        n.Dav_nAt(a, b) = mean( D_nAt{b}(:) .^ 2 );
    end
    
    % Calculate error rates
    if (every == 0) || ( 1 == mod(a, every) )
        fprintf( 'ERROR RATES\n' );
    end
    % Check recognition
    [~,max_id] = max( D_tst{end} );
    [~,min_id] = min( D_tst{end} );
    n.err(a,1) = 1 - sum( digit_tst == max_id )/n.tst;
    n.err(a,3) = 1 - sum( digit_tst == min_id )/n.tst;
    [~,max_id] = max( D_nAt{end} );
    [~,min_id] = min( D_nAt{end} );
    n.err(a,2) = 1 - sum( digit_tst == max_id )/n.tst;
    n.err(a,4) = 1 - sum( digit_tst == min_id )/n.tst;
    if ( every <= 1 ) || ( 1 == mod(a, every) )
        fprintf( [n.errleg{1}, ': ', num2str(n.err(a,1)), '\n' ] );
        fprintf( [n.errleg{2}, ': ', num2str(n.err(a,2)), '\n' ] );
    end
    toc;
end
fprintf('-----\n\tRUN DATA\n');
fprintf('\nTotal epochs          : %d', n.epoch    );
fprintf('\nEpoch size            : %d/%d', n.lrn, size( n.img_lrn, 2) );
fprintf('\nNumber of layers      : %d', n.L_no     );
fprintf('\nSteepness             : %d', n.alpha    );
fprintf('\nSimulation runtime    : %d', n.sim_time );
fprintf('\nSimulation step time  : %d', n.sim_step );
fprintf('\nTime constant         : %d', n.tao );
fprintf('\nWeight initial scaler : %d', n.init_k   );
fprintf('\nMomentum              : %d->(%d steps)->%d', n.mom(1), n.mom(3), n.mom(2));
fprintf('\nLearning rate         : %d', n.learn    );
fprintf('\nLearning rate decay   : %d', n.l_dec    );
fprintf('\nWeight decay          : %d', n.decay    );
fprintf('\nRecorded learn ID     : %d', n.rec_lrn  );
fprintf('\nRecorded test ID      : %d', n.rec_tst  );
fprintf('\nA weights are ID?     : ');
if n.fix_end; fprintf('True'); else; fprintf('False'); end
fprintf('\nB learning on?        : ');
if n.b_learn; fprintf('True\n'); else; fprintf('False\n'); end
fprintf( '\n-----\n' );
end

