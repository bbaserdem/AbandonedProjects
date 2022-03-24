function o = cell_prod_sum(u, v)
%CELL_PROD Elementwise multiply, and sum a matrix.
o = sum([u{:}]) .* v;
end

