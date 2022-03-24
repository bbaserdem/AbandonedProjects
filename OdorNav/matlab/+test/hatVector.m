% Hat vector testing script
clear('all'); %#ok<CLALL>
close('ALL');
rng(0);

% PARAMETERS
N = 3;
NF = 7;
NB = 10;
INT = .33;
LEN = 1;
COL = aux.get_rainbow(NF);

% Generate fluid points that are all interacting
pts = LEN * rand(NF,2);
PTS = Particles(pts);
% Generate border particles along one edge
bdr = LEN * [...
    (1:NB)', zeros(NB,1); ...
    (1:NB)', -ones(NB,1)] / NB;
BDR = Particles(bdr);

% Generate object
A = XSPH('Test hat vectors', PTS, BDR);
A.h = INT;
A.chooseKernel('wendland');
SUP = A.h * A.s;

for it = 0:(N-1)
    % Move points randomly
    A.fluid.r = LEN * rand(NF,2);
    % Calculate distances
    A.calculateDistance;
    
    %-----FIGURE 1: Particle to particle connections
    figure(2*it + 1);
    % Prediction
    subplot(1,2,1);
    x1 = A.fluid.r(A.Fi_ij,1);
    x2 = A.fluid.r(A.Fj_ij,1);
    y1 = A.fluid.r(A.Fi_ij,2);
    y2 = A.fluid.r(A.Fj_ij,2);
    aux.draw_ptsRadii( A.fluid.r, SUP );
    hold('on');
    plot( [x1,x2]', [y1,y2]', '.-k' );
    hold('off');
    setAxesLocal(LEN, SUP, NB);
    title('Predicted: i \rightarrow j connections');
    % Vector
    disp = A.Fr_ij .* A.Fe_ij;
    subplot(1,2,2);
    ind = sub2ind(disp.sdim, A.Fi_ij, A.Fj_ij);
    dx = disp.data{1}(ind);
    dy = disp.data{2}(ind);
    x1 = A.fluid.r(A.Fi_ij,1);
    y1 = A.fluid.r(A.Fi_ij,2);
    x2 = x1 - dx;
    y2 = y1 - dy;
    aux.draw_ptsRadii( A.fluid.r, SUP );
    hold('on');
    plot( [x1,x2]', [y1,y2]', '.-k' );
    hold('off');
    setAxesLocal(LEN, SUP, NB);
    title('Calculated: i \rightarrow j connections');
    
    %-----FIGURE 1: Wall to particle connections
    figure(2*it + 2);
    % Prediction
    subplot(1,2,1);
    x1 = A.fluid.r(A.Bi_ij,1);
    y1 = A.fluid.r(A.Bi_ij,2);
    x2 = A.bound.r(A.Bj_ij,1);
    y2 = A.bound.r(A.Bj_ij,2);
    aux.draw_ptsRadii( A.fluid.r, SUP );
    hold('on');
    aux.draw_ptsRadii( A.bound.r, 0 );
    plot( [x1,x2]', [y1,y2]', '.-k' );
    hold('off');
    setAxesLocal(LEN, SUP, NB);
    title('Predicted: i \rightarrow \mu connections');
    subplot(1,2,2);
    % Vector
    disp = A.Br_ij .* A.Be_ij;
    subplot(1,2,2);
    ind = sub2ind(disp.sdim, A.Bi_ij, A.Bj_ij);
    dx = disp.data{1}(ind);
    dy = disp.data{2}(ind);
    x1 = A.fluid.r(A.Bi_ij,1);
    y1 = A.fluid.r(A.Bi_ij,2);
    x2 = x1 - dx;
    y2 = y1 - dy;
    aux.draw_ptsRadii( A.fluid.r, SUP );
    hold('on');
    aux.draw_ptsRadii( A.bound.r, 0 );
    plot( [x1,x2]', [y1,y2]', '.-k' );
    hold('off');
    setAxesLocal(LEN, SUP, NB);
    title('Calculated: i \rightarrow \mu connections');
    
end

% Axis settings
function setAxesLocal(L,S,N)
xlim([0, L]);
xticks(0:S:L);
ylim([-1/N, L]);
yticks((-floor((1/N)/L)):S:L);
grid('on');
end