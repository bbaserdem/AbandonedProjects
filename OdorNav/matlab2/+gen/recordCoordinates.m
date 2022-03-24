function [SPH] = recordCoordinates(SPH, TIME)
%RECORDCOORDINATES Create a coordinate recording of the SPH simulation

if ~exist('SPH','var')
    SPH = init.lidDrivenCavity(50,10000);
end
if ~exist('NAME','var')
    NAME = ['data/coordinates_', datestr(now,'dd-mm-yyyy_HH:MM:SS')];
end
if ~exist('TIME','var')
    TIME = 30000;
end

% Initialize struct
RES = struct();
RES(TIME).time = [];
RES(TIME).coordinates = [];

% Make sure to finish on errors
try
    for t = 1:TIME
        % Do one step of the simulation
        fprintf('Simulation step, %d\n', t);
        SPH.doTimeStep;
        % Record info
        RES(t).coordinates = SPH.fluid.r;
        RES(t).time = SPH.T;
    end
catch ERR
    warning(ERR.message);
end

save(NAME,'RES');

end

