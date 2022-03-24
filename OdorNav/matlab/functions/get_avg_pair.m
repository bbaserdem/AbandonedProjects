function o = get_avg_pair(u, i, j)
%GET_SUM_pair Get the sum of two things
o = cell(size(u, 2), 1);
k = (1:size(u,1))';
for d = 1:size(u, 2)
    v = (u(i,d)+u(j,d))/2;
    o{d} = sparse([i;j;k], [j;i;k], [v;v;u(:,d)], size(u,1), size(u,1));
end
end