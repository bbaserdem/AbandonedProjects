function [ list ] = indgen( N )
%INDGEN: Generates antisymmetric indices for i<j

k = 1;
list = zeros(N*(N-1)/2, 2);
for i = 1:N
    for j = i+1:N
        list(k,:) = [i,j];
        k = k+1;
    end
end

end

