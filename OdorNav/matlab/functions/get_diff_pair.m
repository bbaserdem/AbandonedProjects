function o = get_diff_pair(u, i, j)
%GET_DIFF_PAIR Get the difference in sparse cell; o{d}(i,j) = u(i,d)-u(j,d) 

o = cell(size(u,2),1);
for d = 1:size(u,2)
    v = u(i,d)-u(j,d);
    o{d} = sparse([i;j], [j;i], [v;-v], size(u,1), size(u,1));
end

end

