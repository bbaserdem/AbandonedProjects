function calculateHash(o)
%CALCULATEHASH Calculates the hash of all simulated particles

%------------------------%
%-----HASH VARIABLES-----%
%------------------------%
% Calculate hash variables if adjacency is cleared
if isempty(o.hsh_vox_adj)
    % Clear particle adjacency due to overhaul
    o.hsh_prt_adj = [];
    % Calculate the expected voxel id range
    firstVox = floor(o.geo_min/o.geo_int);
    lastVox  = floor(o.geo_max/o.geo_int);
    o.hsh_range = lastVox - firstVox + 1;
    o.hsh_offset = firstVox - 1;
    % Get adjacency mapping of voxels
    S = aux.get_adjacent(o.hsh_range);
    o.hsh_vox_adj_mat = S | (S') | logical(speye(o.hsh_vox_num));
    % Create a cell of adjacents
    o.hsh_vox_adj = cell(o.hsh_vox_num, 1);
    for v = 1:o.hsh_vox_num
        o.hsh_vox_adj{v} = find(o.hsh_vox_adj_mat(v,:));
    end
end

%------------------------------%
%-----CALCULATE NEW HASHES-----%
%------------------------------%
% Calculate the hashes of all the particles (non-simulated get 0 as hash)

% Get all simulated particles
sim = o.prt_sim;

% Get the hash-subscript of the simulated particles
vox = zeros(o.prt_num, o.par_dim);
vox(sim,:) = floor(o.prt_pos(sim,:)/o.geo_int) - o.hsh_offset;

% Check if any particle out of bounds; for consistency issues
if any( (vox(sim,:) > o.hsh_range) | (vox(sim,:) < 1), 'all' )
    error('Particle out of bounds');
end

% Get the linear hash
lin = zeros(o.prt_num, 1);
lin(sim,:) = aux.s2i(o.hsh_range, vox(sim,:));

%-------------------------------%
%-----ADJACENCY CALCULATION-----%
%-------------------------------%
% We want to calculate adjacency, either as an <update> or <from scratch>

if isempty(o.hsh_prt_adj)   % CALCULATE FROM SCRATCH
    % Write the values
    o.hsh_prt_vox = lin;
    % Partition particles to their hashes
    [o.hsh_vox_prt, o.hsh_prt_loc] = aux.get_partitions( ...
        o.hsh_prt_vox, o.hsh_vox_num);
    % Get how many particles exist per voxel
    o.hsh_vox_len = cellfun(@(x) length(x), o.hsh_vox_prt);
    % Preallocate the particle adjacency with twice the value for good measure
    n_conn = full( (o.hsh_vox_len') * o.hsh_vox_adj_mat * o.hsh_vox_len );
    o.hsh_prt_adj = logical(spalloc(o.prt_num, o.prt_num, 2*n_conn));
    for v = 1:o.hsh_vox_num         % For each voxel
        for a = o.hsh_vox_adj{v}    % And each of its neighbours
            o.hsh_prt_adj( ...           % Make their particles adjacent
                o.hsh_vox_prt{v}(1:o.hsh_vox_len(v)), ...
                o.hsh_vox_prt{a}(1:o.hsh_vox_len(a)) ) = true;
        end
    end
    
else                        % JUST UPDATE THE VALUES OF CHANGED PARTICLES
    % Find the changed particles
    cPrt = find(lin ~= o.hsh_prt_vox);
    % Get old and new voxels
    oVox = o.hsh_prt_vox(cPrt);
    nVox = lin(cPrt);
    % (!)Overwrite old values
    o.hsh_prt_vox(cPrt) = nVox;
    % Remove the adjacency info of the changed particles
    o.hsh_prt_adj(cPrt,:) = false;
    o.hsh_prt_adj(:,cPrt) = false;
    % Remove/Add particles from/to voxel list
    for n = 1:length(cPrt)
        pc = cPrt(n);
        vo = oVox(n);
        vn = nVox(n);
        % Only remove particle if it was being simulated
        if vo > 0
            % Replace particle in voxel list by the last valid particle
            o.hsh_vox_prt{vo}(o.hsh_prt_loc(pc)) = ...
                o.hsh_vox_prt{vo}(o.hsh_vox_len(vo));
            % Decrease voxel list length
            o.hsh_vox_len(vo) = o.hsh_vox_len(vo) - 1;
        end
        if vn > 0       % Only add particle if it's going to be simulated
            % Increase voxel list length
            o.hsh_vox_len(vn) = o.hsh_vox_len(vn) + 1;
            % Put particle in the end of voxel list
            o.hsh_prt_loc(pc) = o.hsh_vox_len(vn);
            o.hsh_vox_prt{vn}(o.hsh_prt_loc(pc)) = pc;
        else            % If not going to be simulated, remove location info
            o.hsh_prt_loc(pc) = 0;
        end
    end
    % Seperate the changed particles to their new voxel id's (0 excluded)
    [cVoxList, cVox] = aux.get_part_abstract(nVox);
    % Insert connections from changed particles to updated voxels
    for n = 1:length(cVox)      % For every voxel which has incoming particles
        vc = cVox(n);           % Voxel with change
        pn = cPrt(cVoxList{n}); % New particles in this voxel
        va = o.hsh_vox_adj{vc}; % Voxels adjacent to this one
        for a = 1:length(va)    % For every adjacent voxel
            % Get particles in new neighbour voxel (updated)
            pa = o.hsh_vox_prt{va(a)}(1:o.hsh_vox_len(va(a)));
            % Mark changed and adjacent particles as neighbouring
            o.hsh_prt_adj(pn,pa) = true;
            o.hsh_prt_adj(pa,pn) = true;
        end
    end
end

end

