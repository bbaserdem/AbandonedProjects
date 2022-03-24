function [ Figs ] = run_plot( n)
%RUN PLOT plots the output of the network
% Ploting error rates
Xlrn = zeros( size(n.Xhist_lrn{1}, 2), n.L_no  );
Xtst = zeros( size(n.Xhist_lrn{1}, 2), n.L_no  );
XnAt = zeros( size(n.Xhist_lrn{1}, 2), n.L_no  );
Dlrn = zeros( size(n.Xhist_lrn{1}, 2), n.L_no+2);
Dtst = zeros( size(n.Xhist_lrn{1}, 2), n.L_no+2);
DnAt = zeros( size(n.Xhist_lrn{1}, 2), n.L_no+2);
leg = cell(n.L_no+2,1);
for b = 1:n.L_no
    Xlrn(:,b) = mean( n.Xhist_lrn{b  }.^2 );
    Xtst(:,b) = mean( n.Xhist_tst{b,1}.^2 );
    XnAt(:,b) = mean( n.Xhist_tst{b,2}.^2 );
    Dlrn(:,b) = mean( n.Dhist_lrn{b  }.^2 );
    Dtst(:,b) = mean( n.Dhist_tst{b,1}.^2 );
    DnAt(:,b) = mean( n.Dhist_tst{b,2}.^2 );
    leg{b} = ['Layer-',num2str(b)];
end
Dlrn(:,end-1) = mean( n.Dhist_lrn{end  }.^2 );
Dtst(:,end-1) = mean( n.Dhist_tst{end,1}.^2 );
DnAt(:,end-1) = mean( n.Dhist_tst{end,2}.^2 );
D1 = sort( n.Dhist_lrn{end  }.^2 );
D2 = sort( n.Dhist_tst{end,1}.^2 );
D3 = sort( n.Dhist_tst{end,2}.^2 );
Dlrn(:,end) = mean( D1(1:end-1,:) );
Dtst(:,end) = mean( D2(1:end-1,:) );
DnAt(:,end) = mean( D3(1:end-1,:) );
leg{end-1} = ['Layer-',num2str(n.L_no+1)];
leg{end} = 'Last-w/out';

Figs = cell(4,1);

Figs{1} = figure(1);
Figs{1}.WindowStyle = 'docked';
p = plot( -1:0.02:2, n.nlin( -1:0.02:2, n.alpha), '-r', 'LineWidth', 4 );
title( 'Nonlinearity in X' );
ylabel('Output');
xlabel('Input');
axis( [-1, 2, -0.5*n.alpha, 1.5*n.alpha ] );

% X layer averages plots
Figs{2} = figure(2);
Figs{2}.WindowStyle = 'docked';
subplot(2,3,1);
p = semilogy( (0:n.sim_step:n.sim_time), Xlrn, 'LineWidth', 2 );
title( ['X learn id: ', num2str(n.rec_lrn)] );
ylabel( 'Avg X^2' );
xlabel( 'Time' );
if n.L_no ~= 1
    legend(leg{1:end-2}, 'Location', 'southeast');
end
subplot(2,3,2);
p = semilogy( (0:n.sim_step:n.sim_time), Xtst, 'LineWidth', 2 );
title( ['X test id: ', num2str(n.rec_tst)] );
xlabel( 'Time' );
if n.L_no ~= 1
    legend(leg{1:end-2}, 'Location', 'southeast');
end
subplot(2,3,3);
p = semilogy( (0:n.sim_step:n.sim_time), XnAt, 'LineWidth', 2 );
title( ['X (no-A) test id: ', num2str(n.rec_tst)] );
xlabel( 'Time' );
if n.L_no ~= 1
    legend(leg{1:end-2}, 'Location', 'southeast');
end
% D layer averages plots
subplot(2,3,4);
p = semilogy( (0:n.sim_step:n.sim_time), Dlrn, 'LineWidth', 2 );
p(end).LineStyle = ':';
title( ['D learn id: ', num2str(n.rec_lrn)] );
ylabel( 'Avg D^2' );
xlabel( 'Time' );
legend(leg, 'Location', 'southeast');
subplot(2,3,5);
p = semilogy( (0:n.sim_step:n.sim_time), Dtst, 'LineWidth', 2 );
p(end).LineStyle = ':';
title( ['D test id: ', num2str(n.rec_tst)] );
xlabel( 'Time' );
legend(leg, 'Location', 'southeast');
subplot(2,3,6);
p = semilogy( (0:n.sim_step:n.sim_time), DnAt, 'LineWidth', 2 );
p(end).LineStyle = ':';
title( ['D (no-A) test id: ', num2str(n.rec_tst)] );
xlabel( 'Time' );
legend(leg, 'Location', 'southeast');

recog = n.err(:,1:2);
Figs{4} = figure(4);
Figs{4}.WindowStyle = 'docked';
if any( n.err(:,1:2) == 0)
    p = plot( recog );
else
    p = semilogy( recog );
end
p(1).LineWidth = 2;
p(2).LineWidth = 3;
legend( n.errleg{1:size(recog,2)} );
title(  'Recognition error per epoch' );
ylabel( 'Error percentage' );
xlabel( 'Epoch no' );

% X layer global averages plots
Figs{3} = figure(3);
Figs{3}.WindowStyle = 'docked';
subplot(2,3,1);
p = semilogy( 1:n.epoch, n.Xav_lrn, 'LineWidth', 2 );
title( 'X-learn' );
ylabel( 'Avg X^2' );
if n.L_no ~= 1
    legend(leg{1:n.X_no}, 'Location', 'southeast');
end
subplot(2,3,2);
p = semilogy( 1:n.epoch, n.Xav_tst, 'LineWidth', 2 );
title( 'X-test' );
xlabel( 'Epochs' );
if n.L_no ~= 1
    legend(leg{1:n.X_no}, 'Location', 'southeast');
end
subplot(2,3,3);
p = semilogy( 1:n.epoch, n.Xav_nAt, 'LineWidth', 2 );
title( 'X-test-no A' );
if n.L_no ~= 1
    legend(leg{1:n.X_no}, 'Location', 'southeast');
end
% D layer global averages plots
subplot(2,3,4);
p = semilogy( 1:n.epoch, n.Dav_lrn, 'LineWidth', 2 );
title( 'D-learn' );
ylabel( 'Avg D^2' );
legend(leg, 'Location', 'southeast');
subplot(2,3,5);
p = semilogy( 1:n.epoch, n.Dav_tst, 'LineWidth', 2 );
title( 'D-test' );
xlabel( 'Epochs' );
legend(leg{1:n.D_no}, 'Location', 'southeast');
subplot(2,3,6);
p = semilogy( 1:n.epoch, n.Dav_tst, 'LineWidth', 2 );
title( 'D-test-no A' );
legend(leg, 'Location', 'southeast');

end
