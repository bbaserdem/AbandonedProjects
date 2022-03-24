function doQuantities(o)
%DOQUANTITIES Calculate the calculable quantities

% Get distances and kernel
o.calculateDistance;
% Extrapolate density and pressure
o.calculateFluid;
% Get the fluid boundary pressure and density
o.calculateBoundary;
% Calculate the effective forces
o.calculateForce;

end

