% Hat vector testing script
clear('all'); %#ok<CLALL>
close('ALL');

% Iterations;
I = 5;
% Seed rng
rng(0);

% Generate object
A = init.randomPoints(7);

for i = 1:I
    % Move points randomly
    A.prt_pos = rand(A.prt_num,2);
    % Calculate the hash
    A.calculateHash;
    A.calculateDistance;
    % Create new figure
    figure(i+37);
    % Show the hash adjacency
    subplot(2,2,[1,3]);
    A.plotParticles;
    A.plotGrid;
    A.plotInteraction('all', 'all', 'nbr');
    % Show the distance adjacency
    subplot(2,2,2);
    A.plotParticles;
    A.plotGrid;
    A.plotInteraction('all', 'all', 'int');
    A.plotIntCircles;
    % Draw half way arrows for each connections
    subplot(2,2,4);
    A.plotParticles;
    A.plotGrid;
    A.plotVectors( A.cal_int_j, .5 * A.cal_dist(A.cal_int_ij) .* [ ...
        full(A.cal_ehat.data{1}(A.cal_int_ij)), ...
        full(A.cal_ehat.data{2}(A.cal_int_ij))], [1,1,0], 'off' );
    
end