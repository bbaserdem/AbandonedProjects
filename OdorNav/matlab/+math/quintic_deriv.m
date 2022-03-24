function out = quintic_deriv(x, d)
%QUINTIC_DERIV Return quintic kernel derivativefor the given distance vector

switch d
    case 2
        FAC = 7/(pi*478);
    otherwise
        error('Not configured');
end

out = -5 * FAC * (...
    15 * (x<1) .* ((1-x).^4) + ...
    -6 * (x<2) .* ((2-x).^4) + ...
     1 * (x<3) .* ((3-x).^4) );

end

