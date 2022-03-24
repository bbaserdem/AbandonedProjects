function o = get_avg_har(u, w, i, j)
%GET_AVG_HAR Get the harmonic average of two quantities
v = (2*u(i).*w(j))./(u(i)+w(j));
o = sparse(i, j, v, size(u,1), size(w,1));

end