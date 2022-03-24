% Test script
% ----- Plotting is fixed, but the distances algorithm is still bugged
clear
flg = false;

% Build point grid
s = 20;
t = .25;
[x,y] = meshgrid( 0:(t/3):(s-t/3), 0:(t/3):(s-t/3) );
pert = @(u) .1 * t * rand( length(u(:)), 1 ) + u(:);
C = [pert(x), pert(y)];

% Initialize two instances
o1 = loc_SPH( t , [s,s], C );
o2 = loc_SPH( t , [s,s], C );

% Brute force vs fast
tic;
o1.pairwiseDistBrute;
fprintf( 'Time for brute-force: ' );
toc;
tic;
o2.pairwiseDist;
fprintf( 'Time for fast-algor.: ' );
toc;

% Plot
if flg && ( length(x) <= ( 2 ^ 12 ) )
    clf;
    figure;
    subplot(1,2,1);
    o1.plot;
    title( 'Brute-force' );
    subplot(1,2,2);
    o2.plot;
    title('Short calculation');
end
clear ans