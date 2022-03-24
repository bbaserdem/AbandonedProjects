function [ U, A ] = alex_unique_abu( M )
%ALEX_UNIQUE_ABU Find unique rows in M and abundances
%   Sorts M in rows, and stores them in U (I keeps the original index)
%   Find dsM such that if a shift happens from N->N+1, the Nth row of dsM
%   is nonzero. Then sums over dimensions (columns) so that any change is
%   retained (if original dsM = [0 0 8 ...] signifying change from 3 to 4,
%   then ind becomes find([1 0 0 8 . . . = [1 4 ...thus uM extracts the 
%   first row of each cluster)
%   Finds the location of changes by adding 1 to end/front and checking !=0
%   Then uM gets the first row of each cluster that is unique
%   L is a cell of the size of the unique rows
%   L's elements get the list of 


    U = sortrows(M);
    
    % Find points just before state transitions
    drU = U(2:end,:) - U(1:(end-1),:);
    dU = sum(abs(drU),2);
    % Get index before state transitions
    sto = find(dU);
    % Get index where a state transition happened
    dUind = [1; sto+1; size(U,1)+1];
    % Get the index of the new state
    U = U(dUind(1:(end-1)),:);
    % Get the length of state
    A = dUind(2:end) - dUind(1:(end-1));
end
