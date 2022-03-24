function out = wendland_deriv(x, d)
%WENDLAND_DERIV Return wendland kernel derivative for the given distance vector

switch d
    case 1
        FAC = 3/4;
    case 2
        FAC = 7/(4*pi);
    case 3
        FAC = 21/(16*pi);
    otherwise
        error('Not configured');
end

out = FAC .* (x<2).*((1-x/2).^4).*(1+2*x);

end

