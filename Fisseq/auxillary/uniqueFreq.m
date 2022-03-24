function [UQ,freq,address] = uniqueFreq(M, varargin)
%UNIQUEFREQ Unique function that sorts with frequency
%   Detailed explanation goes here

rowF = false;
for a = 1:length(varargin)
    if strcmp('rows',varargin{a})
        rowF = true;
        break;
    end
end

% Get results
if rowF
    UQ = unique(M, 'rows');
else
    UQ = unique;
end
N = size(UQ,1);

% Score for occurence for sorting
score = zeros( N, 1);
address = cell(N,1);
if rowF
    for a = 1:N
        address{a} = find( all( M == UQ(a,:), 2 ) );
        score(a) = length( address{a} );
    end
else
    for a = 1:N
        address{a} = find( M == UQ(a) );
        score(a) = length( address{a} );
    end
end

[freq,I] = sort( score, 'descend' );
address = address(I);
UQ = UQ(I,:);

end

