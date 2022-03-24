function calculateHash(o)
%CALCULATEHASH Calculate the grid; in this case the filters

% Calculate filters if they are empty
if isempty(o.filt_ker)
    % Get grid length
    o.grid_len = o.geo_int / o.grid_sub;
    % Get grid size
    o.grid_dim = ceil( (o.geo_max-o.geo_min) / o.grid_len );
    % Get radii filter
    gen = o.grid_len * (((-o.grid_sub):(o.grid_sub))');
    filt_x = ones(1,length(gen)) .* gen;
    filt_y = ones(length(gen),1) .* (gen');
    % Get radius filter
    o.filt_rad = sqrt( (filt_x.^2) + (filt_y.^2) );
    filt_ex = filt_x ./ (1e-10 + o.filt_rad);
    filt_ey = filt_y ./ (1e-10 + o.filt_rad);
    % Get kernel filters
    o.filt_ker = o.int_kf(o.filt_rad);
    filt_kg = o.int_kd(o.filt_rad);
    o.filt_ker_grad = cell(2,1);
    o.filt_ker_grad{1} = filt_kg .* filt_ex;
    o.filt_ker_grad{2} = filt_kg .* filt_ey;
    o.filt_ker_grad_ten = cell(2,2);
    o.filt_ker_grad_ten{1,1} = o.filt_ker_grad{1} .* filt_ex ./ o.filt_rad;
    o.filt_ker_grad_ten{1,2} = o.filt_ker_grad{1} .* filt_ey ./ o.filt_rad;
    o.filt_ker_grad_ten{2,1} = o.filt_ker_grad{2} .* filt_ex ./ o.filt_rad;
    o.filt_ker_grad_ten{2,2} = o.filt_ker_grad{2} .* filt_ey ./ o.filt_rad;
    % Precalculate some images
    o.pre_kgt = cell(2,2);
    o.pre_kgt{1,1} = imfilter(ones(o.grid_dim), -o.filt_ker_grad_ten{1,1}, 'same');
    o.pre_kgt{1,2} = imfilter(ones(o.grid_dim), -o.filt_ker_grad_ten{1,2}, 'same');
    o.pre_kgt{2,1} = imfilter(ones(o.grid_dim), -o.filt_ker_grad_ten{2,1}, 'same');
    o.pre_kgt{2,2} = imfilter(ones(o.grid_dim), -o.filt_ker_grad_ten{2,2}, 'same');
end

% Get all grid points and record
o.hsh_prt_vox = ceil( (o.prt_pos-o.geo_min)/o.grid_len );
o.hsh_prt_loc = SPH.getS2I(o.grid_dim, o.hsh_prt_vox);

% Create the mass image
o.grid_mas = sparse(o.hsh_prt_vox(:,1), o.hsh_prt_vox(:,2), o.par_mas, ...
    o.grid_dim(1), o.grid_dim(2));

% Create velocity image
o.grid_vel = cell(1,2);
o.grid_vel{1} = sparse(o.hsh_prt_vox(:,1), o.hsh_prt_vox(:,2), ...
    o.prt_vel(:,1), o.grid_dim(1), o.grid_dim(2));
o.grid_vel{2} = sparse(o.hsh_prt_vox(:,1), o.hsh_prt_vox(:,2), ...
    o.prt_vel(:,2), o.grid_dim(1), o.grid_dim(2));

end

