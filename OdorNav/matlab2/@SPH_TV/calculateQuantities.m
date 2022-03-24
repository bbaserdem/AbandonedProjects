function calculateQuantities(o)
%CALCULATEQUANTITIES Calculates system quantities (This is SPH scheme)

%----FLUID PARTICLE QUANTITIES
f_mom = o.prt_mom(o.prt_fld,:);
f_mas = o.prt_mas(o.prt_fld,:);
k_fld_all = o.cal_kern(o.prt_fld,:);
%--Volume
f_vol = 1 ./ full(sum(k_fld_all, 2));
%--Density
f_den = f_mas ./ f_vol;
%--Pressure
f_pre = o.int_pre(f_den);
% Write
o.prt_vol(o.prt_fld,:) = f_vol;
o.prt_den(o.prt_fld,:) = f_den;
o.prt_pre(o.prt_fld,:) = f_pre;

%----BOUNDARY PARTICLE QUANTITIES
b_vel = o.prt_vel(o.prt_bdr,:);
b_mas = o.prt_mas(o.prt_bdr,:);
k_bdr_fld = o.cal_kern(o.prt_bdr,o.prt_fld);
cor = full(sum(k_bdr_fld, 2));
%--Momenta
b_mom = 2*b_vel - full((k_bdr_fld*f_mom)./cor);
%--Pressure is straight averaged pressure with no volume weighting
b_pre = full(k_bdr_fld*f_pre) ./ cor;
%--Density
b_den = o.int_den(b_pre);
%--Volume
b_vol = b_mas ./ b_den;
% Write
o.prt_mom(o.prt_bdr,:) = b_mom;
o.prt_pre(o.prt_bdr,:) = b_pre;
o.prt_den(o.prt_bdr,:) = b_den;
o.prt_vol(o.prt_bdr,:) = b_vol;

%----FORCES ON FLUID PARTICLES
% Prefactor
d_fld_all = o.cal_kgrd(o.prt_fld,:);
pre = ( (f_vol.^2) + ((o.prt_vol').^2) ) .* d_fld_all;
[i, j] = find(d_fld_all);
% E-hat
e_fld_all = o.cal_ehat.data;
for k = 1:length(e_fld_all(:))
    e_fld_all{k} = e_fld_all{k}(o.prt_fld,:);
end
e_fld_all = spTens(e_fld_all);
%--PRESSURE
% Symmetric average
p = sparse(i, j, ...
    (o.prt_den(i,:).*o.prt_pre(j,:)+o.prt_pre(i,:).*o.prt_den(j,:))./...
    (o.prt_den(i,:)+o.prt_den(j,:)), o.prt_fld_num, o.prt_num );
f_frc_pre = sum( (-p) .* pre .* e_fld_all );
%--Advection
% Advection tensor rho*v(v_evil - v)
a = o.prt_mom .* o.prt_den .* ...
    reshape((o.prt_vel-o.prt_mom), o.prt_num, 1, o.par_dim);
aij = spTens(i, j, .5*(a(i,:,:)+a(j,:,:)), o.prt_fld_num, o.prt_num);
f_frc_adv = sum( pre .* (aij * e_fld_all) );
%--Viscosity
n = sparse(i, j, ...
    2*(o.prt_den(i,:).*o.prt_fri(i,:).*o.prt_den(j,:).*o.prt_fri(j,:))./...
    (o.prt_den(i,:).*o.prt_fri(i,:)+o.prt_den(j,:).*o.prt_fri(j,:)), ...
    o.prt_fld_num, o.prt_num);
v = spTens(i, j, o.prt_mom(i,:) - o.prt_mom(j,:), o.prt_fld_num, o.prt_num);
r_fld_all = o.cal_dist(o.prt_fld,:);
f_frc_vis = sum( (pre.*n./r_fld_all) .* v );
%--Correction
f_frc_cor = (-o.par_pre) * sum( pre .* e_fld_all );
% Write
o.prt_for_pre(o.prt_fld,:) = f_frc_pre;
o.prt_for_vis(o.prt_fld,:) = f_frc_vis;
o.prt_for_adv(o.prt_fld,:) = f_frc_adv;
o.prt_for_cor(o.prt_fld,:) = f_frc_cor;


end

