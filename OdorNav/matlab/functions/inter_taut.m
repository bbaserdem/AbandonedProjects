function o = inter_taut(rho, P, R, G, C)
%INTER_TAUT Taut equation (for pressure calculation)
%   P: Referance pressure
%   R: Referance density (rho)
%   G: Gamma
%   C: Constant

o = P * ( (rho/R).^G - 1 ) + C;

end

