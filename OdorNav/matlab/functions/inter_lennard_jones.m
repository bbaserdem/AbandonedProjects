function o = inter_lennard_jones(r, r0, e1, e2, d)
%INTER_LENNARD_JONES Boundary particle repulsion force
%   r0: Referance radius
%   e1: Greater exponential
%   e2: Lesser  exponential
%   d:  Referance force

r = r ./ r0;
o = -d * max(0, (r.^(-e1))-(r.^(-e2)));

end

