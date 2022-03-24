function pre = tait(rho, rho0, c, gamma)
%TAIT Tait equation for density calculation
p0 = gamma * c * c ./ rho0;
pre = p0 * ( (rho./rho0).^gamma - 1 );
end