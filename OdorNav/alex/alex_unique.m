function [ uM, L ] = alex_unique( M )

%ALEX_UNIQUE Find unique rows in M and returns sorted elements uM and recurrent indices L
%   Sorts M in rows, and stores them in sM (I keeps the original index)
%   Find dsM such that if a shift happens from N->N+1, the Nth row of dsM
%   is nonzero. Then sums over dimensions (columns) so that any change is
%   retained (if original dsM = [0 0 8 ...] signifying change from 3 to 4,
%   then ind becomes find([1 0 0 8 . . . = [1 4 ...thus uM extracts the 
%   first row of each cluster)
%   Finds the location of changes by adding 1 to end/front and checking !=0
%   Then uM gets the first row of each cluster that is unique
%   L is a cell of the size of the unique rows
%   L's elements get the list of 


    [sM,I] = sortrows(M);
    
    %revI = I;
    %revI(I)=[1:length(I)]';
    
    %sM
    
    dsM = sM(2:end,:) - sM(1:(end-1),:);
    dsM = sum(abs(dsM),2);
    
    ind=find([double(1); dsM; double(1)]);
    
    uM=sM(ind(1:(length(ind)-1)),:);

    L = cell(size(uM,1),1);
    lll = zeros(size(L));
    
    for i=1:size(uM,1)
        
        i1 = ind(i);
        i2 = ind(i+1)-1;
        
        L{i} = sort(I(i1:i2), 'ascend');
        lll(i)=length(L{i});
        
    end
    
    [slll,ind]=sort(lll,'descend');
    
    %L=L(ind);
    %uM = uM(ind,:);
end
