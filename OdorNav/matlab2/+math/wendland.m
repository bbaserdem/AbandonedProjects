function out = wendland(x, d)
%WENDLAND Return wendland kernel for the given distance vector

% thr can be a vector(same as dist) or scalar

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

out = FAC .* (x<2).*((1-x/2).^3).*(-5*x);

end

