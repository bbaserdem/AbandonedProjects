function [ R, grad ] = Rgen( param, N )
%Rgen Generates a SO matrix from given parameters and the individual
%gradient matrices

% Generate index pairs
index = indgen(N);
K = length(param)
if K ~= length(index)
    error('Wrong # of parameters');
end

R = eye(N);
grad = cell(1,K);
grad{:} = eye(N);

for i = 1:K
    store = rot_gen( param(i), index(i,:), N);
    R = R * store;
    grad{1:i-1} = grad{1:i-1} * store;
    grad{i} = grad{i} * drot_gen( param(i), index(i,:), N);
    grad{i+1:K} = grad{i+1:K} * store;
end

end