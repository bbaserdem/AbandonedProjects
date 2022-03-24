function oo = minus(o, b)
%MINUS Overload the subtraction function

if isobject(b) && isobject(o)
    % Get sizes of both objects
    oL = o.dim;
    bL = b.dim;
    % Make size vectors the same lenght (to repeat dimensions) by expanding
    % the length matrix with ones;
    if length(oL) > length(bL)
        bL = [bL, ones(1, length(oL)-length(bL))];
    elseif length(oL) < length(bL)
        oL = [oL, ones(1, length(bL)-length(oL))];
    end
    % Interpolate size of output tensor
    dL = max(oL,bL);
    % Get list of indices to calculate in the tensors
    dS = aux.i2s(dL, (1:prod(dL))');
    oS = min(oL, dS);
    bS = min(bL, dS);
    % Convert these indices to linear
    oI = aux.s2i(oL, oS);
    bI = aux.s2i(bL, bS);
    % Do the summation of the cell
    d = cell(dL);
    for i = 1:prod(dL)
        d{i} = o.data{oI(i)} - b.data{bI(i)};
    end
elseif isobject(b)
    if ~issparse(o)
        warning('Full array addition to SparseTensor object is very inefficient!');
    end
    d = cellfun(@(x) sparse(o-x), b.data, 'UniformOutput', false);
else
    if ~issparse(b)
        warning('Full array addition to SparseTensor object is very inefficient!');
    end
    d = cellfun(@(x) sparse(x-b), o.data, 'UniformOutput', false);
end

oo = SparseTensor(d);

end
