function [pts, bnd] = gen_box_particles(num, len, dim, pack)
%GEN_BOX_PARTICLES Generate coordinates of particles in a box
%#ok<*AGROW>

% Sane defaults
if ~exist('num','var')
    num = 40;
end
if ~exist('len','var')
    len = 1;
end
if ~exist('dim','var')
    dim = 2;
end
if ~exist('pack','var')
    pack = 2;
end

p_gen = ((len/2)/(num*pack+1))*(linspace(pack+1, (2*num-1)*pack+1, num)');
b_gen = linspace(0, len, pack*num+2)';
c_gen = linspace(0, len, 2)';

pts = zeros(1,0);
bnd = zeros(0,0);
cap = zeros(1,0);

for d = 1:dim
    % Extend point in new dimension, (put in on the first one)
    pts = [ repelem(pts, num, 1), repmat(p_gen, size(pts,1), 1)];
    % Extend boundaries in new dimension
    bnd = [ repelem(bnd, pack*num+2, 1), repmat(b_gen, size(bnd,1), 1)];
    % Append the caps to close dimension
    bnd = [bnd; repelem(cap, 2, 1), repmat(c_gen, size(cap,1), 1)]; 
    % Extend the cap
    cap = [repelem(cap, pack*num, 1), ...
        repmat(b_gen(2:(end-1)), size(cap,1), 1)];
end


end

