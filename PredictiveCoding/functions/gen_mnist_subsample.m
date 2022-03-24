function [ train_img, train_lbl, ...
    test_img, test_lbl ] = gen_mnist_subsample( inp_dig )
%GEN_MNIST_SUBSAMPLE Generates a MNIST sumsample dataset of digits

load MNIST_cell.mat

% Input checking
inp_dig = inp_dig(:);
if (length(inp_dig) > 10) || any( (inp_dig < 1) | (inp_dig > 10) )
    error('Invalid input');
end

train_img = [];
train_lbl = [];
test_img = [];
test_lbl = [];

for a = 1:length(inp_dig)
    b = inp_dig(a);
    train_img = [ train_img, mnist_60k{b,1} ];
    train_lbl = [ train_lbl, mnist_60k{b,2}*ones(1,size(mnist_60k{b,1},2))];
    test_img = [ test_img, mnist_10k{b,1} ];
    test_lbl = [ test_lbl, mnist_10k{b,2}*ones(1,size(mnist_10k{b,1},2))];
end

perm_train = randperm( size(train_img,2) );
perm_test = randperm( size(test_img,2) );

train_img = train_img(:,perm_train);
train_lbl = train_lbl(:,perm_train);
test_img = test_img(:,perm_test);
test_lbl = test_lbl(:,perm_test);
end

