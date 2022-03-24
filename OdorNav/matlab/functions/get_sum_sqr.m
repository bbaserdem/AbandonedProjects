function o = get_sum_sqr(u, w, i, j)
%GET_SUM_SQR_SELF Get the sum of two things
v = (u(i).^2)+(w(j).^2);
o = sparse(i, j, v, size(u,1), size(w,1));
end