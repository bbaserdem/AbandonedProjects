function o = get_diff_uv(u, v, i, j)
%GET_DIFF_UV Get the difference in sparse cell; o{d}(i,j) = u(i,d)-v(j,d) 

o = cell(size(u,2),1);
for d = 1:size(u,2)
    o{d} = sparse(i, j, u(i,d)-v(j,d), size(u,1), size(v,1));
end

end

