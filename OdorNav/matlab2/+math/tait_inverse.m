function rho = tait_inverse(pre, p0, c, gamma)
%TAIT Tait equation for density calculation
rho0 = gamma * c * c ./ p0;
rho = (((pre./p0)+1).^(1/gamma)).*rho0;
end