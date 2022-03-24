function o = get_avg_weight(u, w, uu, ww, i, j)
%GET_AVG_WEIGHT Get the average, weighted by a factor
v = (u(i).*ww(j)+w(i).*uu(j))./(w(i)+ww(j));
o = sparse(i, j, v, size(u,1), size(uu,1));

end