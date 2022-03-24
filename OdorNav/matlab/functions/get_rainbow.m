function out = get_rainbow(N, sat)
% GET_RAINBOW Function that returns colors in HSV space
if ~exist('sat','var')
    sat = 1;
end
if isscalar(sat)
    sat = ones(1,N)*sat;
elseif isvector(sat)
    if size(sat, 1) ~= 1
        sat = sat';
    end
end
out = hsv2rgb([linspace(0,1,N); sat; ones(1,N)]');
end

