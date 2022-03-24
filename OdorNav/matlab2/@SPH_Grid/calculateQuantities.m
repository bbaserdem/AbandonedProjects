function calculateQuantities(o)
%CALCULATEQUANTITIES Calculates system quantities

% Extrapolate continuos density;
o.grid_den = imfilter(full(o.grid_mas), -o.filt_ker, 'same') / o.par_vol;

% Extrapolate pressure
o.grid_pre = sparse(o.hsh_prt_vox(:,1), o.hsh_prt_vox(:,2), ...
    o.int_pre(o.grid_den(o.hsh_prt_loc)), o.grid_dim(1), o.grid_dim(2));

% Calculate pressure force;
%---Naive= G_i P_a = sum_b Pb *|* W'_ab e_i_ab
o.grid_for_pre = cell(1,2);
o.grid_for_pre{1} = (-1./o.grid_den) .* ...
    imfilter(full(o.grid_pre), -o.filt_ker_grad{1}, 'same');
o.grid_for_pre{2} = (-1./o.grid_den) .* ...
    imfilter(full(o.grid_pre), -o.filt_ker_grad{2}, 'same');

% Calculate viscous force;
o.grid_for_vis = cell(1,2);
% For the first velocity component
o.grid_for_vis{1} = 8 * o.par_vis * o.par_vol * ( ...
    o.grid_vel{1} .* o.pre_kgt{1,1} + o.grid_vel{2} .* o.pre_kgt{2,1} + ...
    -imfilter(full(o.grid_vel{1}), -o.filt_ker_grad_ten{1,1}) + ...
    -imfilter(full(o.grid_vel{2}), -o.filt_ker_grad_ten{2,1}) );
% For the second velocity component
o.grid_for_vis{2} = 8 * o.par_vis * o.par_vol * ( ...
    o.grid_vel{1} .* o.pre_kgt{1,2} + o.grid_vel{2} .* o.pre_kgt{2,2} + ...
    -imfilter(full(o.grid_vel{1}), -o.filt_ker_grad_ten{1,2}) + ...
    -imfilter(full(o.grid_vel{2}), -o.filt_ker_grad_ten{2,2}) );

% Get forces
id_fld = o.hsh_prt_loc(o.prt_fld);
o.prt_for_vis(o.prt_fld,1) = full(o.grid_for_vis{1}(id_fld));
o.prt_for_vis(o.prt_fld,2) = full(o.grid_for_vis{2}(id_fld));
o.prt_for_pre(o.prt_fld,1) = full(o.grid_for_pre{1}(id_fld));
o.prt_for_pre(o.prt_fld,2) = full(o.grid_for_pre{2}(id_fld));

end

