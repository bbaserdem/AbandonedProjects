function o = get_sum(u, w, i, j)
%GET_SUM Get the sum of two things
v = (u(i)+w(j))/2;
o = sparse(i, j, v, size(u,1), size(w,1));
end