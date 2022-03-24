function [ O ] = issquare( M, flag )
%ISSQUARE Check if matrix is square
% Can check for multidim too with variable 'multidim'

O = false;

if ~exist('flag','var')
    flag = 'singledim';
end

if strcmp(flag,'singledim')
    O = size(M,1);
    if all( size(M) == O ) && ( length(size(M)) == 2 )
        O = true;
    else
        O = false;
    end
elseif strcmp(flag, 'multidim')
    O = size(M,1);
    if all( size(M) == O )
        O = true;
    else
        O = false;
    end
else
    error('Invalid flag argument')
end

end

