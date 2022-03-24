function o = get_sum_sqr_pair(u, i, j)
%GET_SUM_SQR_SELF Get the sum of two things
uu = u.^2;
v = uu(i)+uu(j);
k = (1:size(u,1))';
o = sparse([i;j;k], [j;i;k], [v;v;uu], size(u,1), size(u,1));
end