function o = get_avg_weight_pair(u, w, i, j)
%GET_AVG_WEIGHT_PAIR Get the average, weighted by a factor
v = (u(i).*w(j)+w(i).*u(j))./(w(i)+w(j));
k = (1:size(u,1))';
o = sparse([i;j;k], [j;i;k], [v;v;u], size(u,1), size(u,1));

end