function updateAdjacency(o)
%UPDATEADJACENCY Get list of particles that are within interaction range

% Check if boundary particle hashing has been done before
if isempty(o.hashAdj)
    o.initHash;
end

% Get the hash for flruid particles
o.fluid.getHash(o.sup, o.hashOffset, o.hashSubDim, o.hashAdj);

% Initialize some counting stuff
FI = zeros(o.fNum*(o.fNum-1)/2, 1);
FJ = zeros(o.fNum*(o.fNum-1)/2, 1);
BI = zeros(o.fNum*o.bNum, 1);
BJ = zeros(o.fNum*o.bNum, 1);
fn = 0;
bn = 0;

% Loop over all the voxels
for h = 1:o.hashLinDim
    % Get the particles in voxel, and candidates to voxel (fluid)
    vox = o.fluid.parHash{h};
    adj = o.fluid.adjHash{h};
    vn = o.fluid.parLen(h);
    an = o.fluid.adjLen(h);
    % Insert cross voxel ij pairs
    n = vn*an;
    FI((fn+1):(fn+n)) = repelem(vox, 1, an);
    FJ((fn+1):(fn+n)) = repmat( adj, 1, vn);
    fn = fn + n;
    % Insert crosstalk within this voxel
    [i, j] = o.memGetIJ(vn);
    n = vn*(vn-1)/2;
    FI((fn+1):(fn+n)) = vox(i);
    FJ((fn+1):(fn+n)) = vox(j);
    fn = fn + n;
    % Get the particles in voxel, and candidates to voxel (bound)
    voxB = o.bound.parHash{h};
    adjB = o.bound.adjHash{h};
    vnB = o.bound.parLen(h);
    anB = o.bound.adjLen(h);
    % Voxel(border) to Adjacent(fluid) accros voxels
    n = vnB*an;
    BI((bn+1):(bn+n)) = repelem(adj, 1, vnB);
    BJ((bn+1):(bn+n)) = repmat(voxB, 1, an );
    bn = bn + n;
    % Voxel(fluid) to Adjacent(border) accross voxels
    n = vn*anB;
    BI((bn+1):(bn+n)) = repelem(vox, 1, anB);
    BJ((bn+1):(bn+n)) = repmat(adjB, 1, vn );
    bn = bn + n;
    % Insert crosstalk within this voxel
    [i, j] = o.memGetIJ(vn, vnB);
    n = vn*vnB;
    BI((bn+1):(bn+n)) = vox(i);
    BJ((bn+1):(bn+n)) = voxB(j);
    bn = bn + n;
end

% Truncate the index
FI = FI(1:fn);
FJ = FJ(1:fn);
BI = BI(1:bn);
BJ = BJ(1:bn);

% PARTICLE PAIRS
% Calculate pairwise distances only on valid index
e = (o.fluid.r(FI,:)-o.fluid.r(FJ,:));
d = sum(e.^2, 2);
% Check if further valid
v = (d<(o.sup^2)) & (d>0);
% Correct distances
i = FI(v);
j = FJ(v);
d = sqrt(d(v));
e = e(v,:) ./ d;
% Write data
o.Fi_ij = i;
o.Fj_ij = j;
o.Fr_ij = sparse([i;j], [j;i], [d;d], o.fNum, o.fNum);
o.Fe_ij = SparseTensor([i;j], [j;i], [e;-e], o.fNum, o.fNum);

% PARTICLE vs. BOUNDARY
% Calculate pairwise distances only on valid index
e = (o.fluid.r(BI,:)-o.bound.r(BJ,:));
d = sum(e.^2, 2);
% Check if further valid
v = d<(o.sup^2) & (d>0);
% Correct distances
i = BI(v);
j = BJ(v);
d = sqrt(d(v));
e = e(v,:) ./ d;
% Write data
o.Bi_ij = i;
o.Bj_ij = j;
o.Br_ij = sparse(i, j, d, o.fNum, o.bNum);
o.Be_ij = SparseTensor(i, j, e, o.fNum, o.bNum);

% KERNEL FUNCTION
% Record the value for the kernel function
rij = o.Fr_ij ./ o.h;
o.Fw_ij = spfun(@(x) o.kernel(x), rij) + o.kernel(0)*speye(o.fNum);
o.Fg_ij = spfun(@(x) o.kerDer(x), rij);
rij = o.Br_ij ./ o.h;
o.Bw_ij = spfun(@(x) o.kernel(x), rij);
o.Bg_ij = spfun(@(x) o.kerDer(x), rij);

end


















