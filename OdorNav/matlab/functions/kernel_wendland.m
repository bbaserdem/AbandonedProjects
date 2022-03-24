function out = kernel_wendland(dist, thr, dim, der)
%KERNEL_WENDLAND Return wendland kernel for the given distance vector

% thr can be a vector(same as dist) or scalar

switch dim
    case 1
        FAC = 3./(4*(thr.^dim));
    case 2
        FAC = 7./(4*pi*(thr.^dim));
    case 3
        FAC = 21./(16*pi*(thr.^dim));
    otherwise
        error('Not configured');
end

q = dist./thr;
if der
    FAC = FAC ./ thr;
    out = FAC .* (q<2).*((1-q/2).^3).*(-5*q);
else
    out = FAC .* (q<2).*((1-q/2).^4).*(1+2*q);
end

end

