function initHash(o)
%INITHASH Initialize info for hashing

% Get geometry, and set xy limits
cor = [o.fluid.r; o.bound.r];
o.locMax = max(cor, [], 1);
o.locMin = min(cor, [], 1);
o.hashOffset = -1 + floor(o.locMin/o.sup);
o.hashSubDim = floor(o.locMax/o.sup) - o.hashOffset;

% Get the boundaries of hash states (for future plotting)
o.hashTicks = cell(o.dim, 1);
for d = 1:o.dim
    o.hashTicks{d} = ((1:o.hashSubDim(d))+o.hashOffset(d))*o.sup;
end

% Get adjacent partitions
[o.hashAdj, o.hashAdjRev] = aux.get_adjacent(o.hashSubDim);

% Calculate boundary particle hashes
o.bound.getHash(o.sup, o.hashOffset, o.hashSubDim, ...
    o.hashAdj, o.hashAdjRev);

% Calculate fluid particle hashes
o.fluid.getHash(o.sup, o.hashOffset, o.hashSubDim, ...
    o.hashAdj, o.hashAdjRev );
% 
% %----CALCULATE ADJACENCY MAP----%
% 
% % Initialize some counting stuff
% FI = zeros(o.fNum*(o.fNum-1)/2, 1);
% FJ = zeros(o.fNum*(o.fNum-1)/2, 1);
% BI = zeros(o.fNum*o.bNum, 1);
% BJ = zeros(o.fNum*o.bNum, 1);
% fn = 0;
% bn = 0;
% 
% % Loop over all possible voxels
% for h = 1:o.hashLinDim
%     % Get the particles in voxel, and candidates to voxel (fluid)
%     vox = o.fluid.parHash{h};
%     adj = o.fluid.adjHash{h};
%     vn = o.fluid.parLen(h);
%     an = o.fluid.adjLen(h);
%     % Insert cross voxel ij pairs
%     n = vn*an;
%     FI((fn+1):(fn+n)) = repelem(vox, 1, an);
%     FJ((fn+1):(fn+n)) = repmat( adj, 1, vn);
%     fn = fn + n;
%     % Insert crosstalk within this voxel
%     [i, j] = o.memGetIJ(vn);
%     n = vn*(vn-1)/2;
%     FI((fn+1):(fn+n)) = vox(i);
%     FJ((fn+1):(fn+n)) = vox(j);
%     fn = fn + n;
%     % Get the particles in voxel, and candidates to voxel (bound)
%     voxB = o.bound.parHash{h};
%     adjB = o.bound.adjHash{h};
%     vnB = o.bound.parLen(h);
%     anB = o.bound.adjLen(h);
%     % Voxel(border) to Adjacent(fluid) accros voxels
%     n = vnB*an;
%     BI((bn+1):(bn+n)) = repelem(adj, 1, vnB);
%     BJ((bn+1):(bn+n)) = repmat(voxB, 1, an );
%     bn = bn + n;
%     % Voxel(fluid) to Adjacent(border) accross voxels
%     n = vn*anB;
%     BI((bn+1):(bn+n)) = repelem(vox, 1, anB);
%     BJ((bn+1):(bn+n)) = repmat(adjB, 1, vn );
%     bn = bn + n;
%     % Insert crosstalk within this voxel
%     [i, j] = o.memGetIJ(vn, vnB);
%     n = vn*vnB;
%     BI((bn+1):(bn+n)) = vox(i);
%     BJ((bn+1):(bn+n)) = voxB(j);
%     bn = bn + n;
% end
% 
% % Write results in adjacency map
% o.F_adj = [FI(1:fn), FJ(1:fn)];
% o.B_adj = [BI(1:bn), BJ(1:bn)];

end

