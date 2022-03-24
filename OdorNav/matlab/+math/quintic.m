function out = quintic(x, d)
%QUINTIC Return quintic kernel for the given distance vector

switch d
    case 2
        FAC = 7/(pi*478);
    otherwise
        error('Not configured');
end

out = FAC * (...
    15 * (x<1) .* ((1-x).^5) + ...
    -6 * (x<2) .* ((2-x).^5) + ...
     1 * (x<3) .* ((3-x).^5) );

end

