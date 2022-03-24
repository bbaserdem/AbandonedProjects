function doAdjacency(o)
%DOADJACENCY Recalculate adjacency info depending on new coordinates

% Do the hash of the border particles if not done before
if isempty(o.hashAdj)
    o.initHash;
    return;
end

% Recalculate hashes
oldHashFluid = o.fluid.linHash;
o.fluid.getHash(o.sup, o.hashOffset, o.hashSubDim, ...
    o.hashAdj, o.hashAdjRev);

% Find the hashes that have changed
part_hashChanged = find(oldHashFluid-o.fluid.linHash ~= 0);
part_num = length(part_hashChanged);

% If fluid hashes were static, don't recalculate
if part_num == 0
    return
end

% Remove the hash of the changed points
sto = any(o.F_adj(:,1)==(part_hashChanged'), 2) | ...
    any(o.F_adj(:,2)==(part_hashChanged'), 2);
F_sto = o.F_adj(~sto,:);
sto = any(o.B_adj(:,1)==(part_hashChanged'), 2);
B_sto = o.B_adj(~sto,:);

% Calculate the hashes with new particles, and the new particle ID's in
% those hashes
part_newHash = o.fluid.linHash(part_hashChanged);
[sto, hash_newPart] = aux.get_partitions_abstract(part_newHash);
hash_newPartIds = cellfun(@(x) part_hashChanged(x)', sto, ...
    'UniformOutput', false);
hash_num = length(hash_newPartIds);

% Initialize some counting stuff
FI = zeros(part_num*o.fNum, 1);
FJ = zeros(part_num*o.fNum, 1);
BI = zeros(part_num*o.bNum, 1);
BJ = zeros(part_num*o.bNum, 1);
fn = 0;
bn = 0;

% Loop over all the hashes with a new particles in them
for h = 1:hash_num
    
    % Get current hash, and the changed particles
    hash = hash_newPart(h);
    part = hash_newPartIds{h};
    pn = length(part);
    
    % Get fluid particles in this voxel
    vox = o.fluid.parHash{hash};
    vn = o.fluid.parLen(hash);
    
    % Get fluid particles positively adjacent to this voxel
    adjP = o.fluid.adjHash{hash};
    apn = o.fluid.adjLen(hash);
    
    % Get fluid particles negatively adjacent to this voxel
    adjN = o.fluid.adjRHash{hash};
    ann = o.fluid.adjRLen(hash);
    
    % Mix the new particles in this voxel with the voxel members
    % While removing self referance
    sto_i = repelem(part, 1, vn);
    sto_j = repmat(  vox, 1, pn);
    sto = (sto_i ~= sto_j);
    n = sum(sto);
    FI((fn+1):(fn+n)) = sto_i(sto);
    FJ((fn+1):(fn+n)) = sto_j(sto);
    fn = fn + n;
    
    % Mix new particles in this voxel with the + neighbours
    n = pn*apn;
    FI((fn+1):(fn+n)) = repelem(part, 1, apn);
    FJ((fn+1):(fn+n)) = repmat(  adjP, 1, pn);
    fn = fn + n;
    
    % Mix new particles in this voxel with the - neighbours
    n = pn*ann;
    FI((fn+1):(fn+n)) = repelem(part, 1, ann);
    FJ((fn+1):(fn+n)) = repmat(  adjN, 1, pn);
    fn = fn + n;
    
    % Get boundary particles in this voxel
    vox = o.bound.parHash{hash};
    vn = o.bound.parLen(hash);
    
    % Get boundary particles positively adjacent to this voxel
    adjP = o.bound.adjHash{hash};
    apn = o.bound.adjLen(hash);
    
    % Get boundary particles negatively adjacent to this voxel
    adjN = o.bound.adjRHash{hash};
    ann = o.bound.adjRLen(hash);
    
    % Mix the new particles in this voxel with the voxel members
    n = pn* vn;
    BI((bn+1):(bn+n)) = repelem(part, 1, vn);
    BJ((bn+1):(bn+n)) = repmat(  vox, 1, pn);
    bn = bn + n;
    
    % Mix new particles in this voxel with the + neighbours
    n = pn*apn;
    BI((bn+1):(bn+n)) = repelem(part, 1, apn);
    BJ((bn+1):(bn+n)) = repmat(  adjP, 1, pn);
    bn = bn + n;
    
    % Mix new particles in this voxel with the - neighbours
    n = pn*ann;
    BI((bn+1):(bn+n)) = repelem(part, 1, ann);
    BJ((bn+1):(bn+n)) = repmat(  adjN, 1, pn);
    bn = bn + n;
end

% Write the new adjacency list
o.F_adj = [F_sto; [FI(1:fn), FJ(1:fn)] ];
o.B_adj = [B_sto; [BI(1:bn), BJ(1:bn)] ];

end

