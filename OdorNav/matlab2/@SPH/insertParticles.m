function insertParticles(o, prt, type)
%INSERTPARTICLES Insert points into simulation (with default characteristics)

if ~exist('type', 'var')
    type = 'fluid';
end
% Number of particles to add
N = size(prt, 1);

% Get non-simulated particles
id = o.prt_nsm_id;
% If there needs to be extra added particle locations, generate them
id = [id; o.prt_num+((1:(N-length(id)))')];
id = id(1:N);

% Set the particle flags
switch type
    case 'fluid'
        o.prt_fld(id,1) = true;
        o.prt_bdr(id,1) = false;
        o.prt_inl(id,1) = false;
        o.prt_out(id,1) = false;
    case {'boundary', 'border'}
        o.prt_fld(id,1) = false;
        o.prt_bdr(id,1) = true;
        o.prt_inl(id,1) = false;
        o.prt_out(id,1) = false;
    case {'input', 'inlet'}
        o.prt_fld(id,1) = false;
        o.prt_bdr(id,1) = false;
        o.prt_inl(id,1) = true;
        o.prt_out(id,1) = false;
    case {'output', 'outlet'}
        o.prt_fld(id,1) = false;
        o.prt_bdr(id,1) = false;
        o.prt_inl(id,1) = false;
        o.prt_out(id,1) = true;
    otherwise
        error('Invalid particle typing');
end

% Insert the particles
o.prt_pos(id,:) = prt;

% Input sane defaults
o.prt_vel(id,:) = zeros(N, o.par_dim);
o.prt_mom(id,:) = zeros(N, o.par_dim);
o.prt_acc(id,:) = zeros(N, o.par_dim);
o.prt_vol(id,:) = o.par_vol * ones(N, 1);
o.prt_den(id,:) = o.par_den * ones(N, 1);
o.prt_pre(id,:) = o.par_pre * ones(N, 1);
o.prt_fri(id,:) = o.par_vis * ones(N, 1);
o.prt_mas(id,:) = o.par_mas * ones(N, 1);
o.prt_for_pre(id,:) = zeros(N, o.par_dim);
o.prt_for_adv(id,:) = zeros(N, o.par_dim);
o.prt_for_vis(id,:) = zeros(N, o.par_dim);
o.prt_for_ext(id,:) = zeros(N, o.par_dim);
o.prt_for_cor(id,:) = zeros(N, o.par_dim);

end

