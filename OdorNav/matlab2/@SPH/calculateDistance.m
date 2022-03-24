function calculateDistance(o)
%CALCULATEDISTANCE Calculate interparticle distances and related quantities

% Get candidates, eliminate pairs
[i, j] = find(triu(o.hsh_prt_adj,1));

% Get difference vector, from j to i
dif = o.prt_pos(i,:) - o.prt_pos(j,:);
% Get square distances
d2 = sum(dif.^2, 2);

% Mask out interactions that are too long (and are 0)
m = (d2 < (o.geo_int^2)) & (d2 > 0);

% Apply the mask
d = sqrt(d2(m));
e = dif(m,:) ./ d;
i = i(m);
j = j(m);

% Record the values
o.cal_int_i = [i; j];
o.cal_int_j = [j; i];
o.cal_int_ij = o.getS2I(o.prt_num*[1,1], [o.cal_int_i,o.cal_int_j]);
o.cal_dist = sparse(o.cal_int_i, o.cal_int_j, [d;d], o.prt_num, o.prt_num);
o.cal_ehat = spTens(o.cal_int_i, o.cal_int_j, [e;-e], o.prt_num, o.prt_num);

% Calculate kernel
r = o.cal_dist / o.geo_smt;
o.cal_kern = spfun(@(x) o.int_kf(x), r) + o.int_kf(0)*speye(o.prt_num);
o.cal_kgrd = spfun(@(x) o.int_kd(x), r);

end

