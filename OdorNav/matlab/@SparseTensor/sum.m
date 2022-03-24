function D = sum(o)
%SUM Overload the summation function; to contract the second sparse index
%   and return a numeric array.

% If this was scalar anyway, just return the column
if isscalar(o)
    D = full(sum(o.data{1}, 2));
else
    % Auto shrink empty tensorial dimensions
    T = o.dim(o.dim>0);
    % Contract the second index in all the tensor elements
    D = zeros([o.sdim(1), prod(T)]);
    for i = 1:size(D,2)
        D(:,i) = full(sum(o.data{i},2));
    end
    % Restore the shape of the non-zero tensor indices
    D = reshape(D, [o.sdim(1), T]);
end

end

