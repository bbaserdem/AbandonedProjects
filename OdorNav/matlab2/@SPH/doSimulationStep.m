function doSimulationStep(o)
%DOSIMULATIONSTEP Does one step of simulation

o.calculateHash;
o.calculateDistance;
o.calculateQuantities;
o.calculateMovement;
end

