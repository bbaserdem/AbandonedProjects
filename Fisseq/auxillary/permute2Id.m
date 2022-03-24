function [P, Mp] = permute2Id( M, rowFlag )
%PERMUTETOID Permutes a matrix to look like identity matrix
% M:    Square matrix to permute
% Mp:   Permuted matrix
% P:    Permutation

if ~issquare(M)
    error('Needs square matrix');
end

if ~exist('rowFlag','var')
    rowFlag = 'cols';
end

if strcmp(rowFlag, 'rows')
    M = M';
elseif ~strcmp(rowFlag, 'cols')
    error('Invalid second argument')
end

I = eye(size(M,1));
PP = perms(1:size(M,1));
score = zeros(1,size(PP,1));

for p = 1:length(score)
    P = PP(p,:);
    Mp = M(:,P);
    score(p) = sum( ( I(:) - Mp(:) ).^2 );
end
[~,p] = min(score);
P = PP(p,:)';

end