% TESTING THE LID DRIVEN CAVITY CASE
clear('all'); %#ok<CLALL>

% Initialize case, the same as adams paper
A = init.lidDrivenCavity(100, 1000);

% Create video of this simulation
gen.recordVideo(A, 'test_ldc');