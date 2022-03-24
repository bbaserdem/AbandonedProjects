function [I, X] = get_partitions_abstract(L)
%GET_PARTITIONS_ABSTRACT Same as base function, but no empty cell space

% Cast L as uint64
L = uint64(L);

% Get the sorted label list from L
X = unique(L);

% Get the list when indices change in L
[val,ind] = sort(L);
trn = find( val(2:end) - val(1:(end-1)) );
trn = [uint64(0); trn; uint64(length(ind))];

% Partition the id matrix by their index
I = mat2cell(ind', 1, trn(2:end)-trn(1:(end-1)))';

end

