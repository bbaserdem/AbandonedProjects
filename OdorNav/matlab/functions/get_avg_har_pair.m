function o = get_avg_har_pair(u, i, j)
%GET_AVG_HAR_PAIR Get the harmonic average of two quantities
v = (2*u(i).*u(j))./(u(i)+u(j));
k = (1:size(u,1))';
o = sparse([i;j;k], [j;i;k], [v;v;u], size(u,1), size(u,1));

end