function [I, X] = get_partitions(L,M)
%GET_PARTITIONS Partition a label list, into cell array containing the labels.

% Cast L as uint64
L = uint64(L);

X = unique(L);
if ~exist('M','var')
    M = max(L(:));
end

[val,ind] = sort(L);
trn = find( val(2:end) - val(1:(end-1)) );
trn = [uint64(0); trn; uint64(length(ind))];

if length(X) ~= M
    I = cell(M,1);
    for i = 1:length(X)
        I{X(i)} = ind((trn(i)+1):trn(i+1))';
    end
else
    I = mat2cell(ind', 1, trn(2:end)-trn(1:(end-1)))';
end

end

