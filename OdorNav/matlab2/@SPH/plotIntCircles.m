function plotIntCircles(o, want)
%PLOTINTCIRCLES Draw cirles around interaction range

if ~exist('want', 'var')
    want = 'all';
end

c = zeros(o.prt_num,3);
c(o.prt_fld,:) = o.getColors(o.prt_fld_num);
c(o.prt_bdr,:) = ones(o.prt_bdr_num,1) .* [0.75, 0.75, 0.75];
c(o.prt_inl,:) = ones(o.prt_inl_num,1) .* [0.80, 1.00, 0.80];
c(o.prt_out,:) = ones(o.prt_out_num,1) .* [1.00, 0.80, 0.80];

switch want
    case 'all'
        m = o.prt_sim_id;
    case 'fluid'
        m = o.prt_fld_id;
    case {'boundary', 'border'}
        m = o.prt_bdr_id;
    case 'inlet'
        m = o.prt_inl_id;
    case 'outlet'
        m = o.prt_out_id;
    otherwise
        error('Invalid particle type');
end
p = o.prt_pos(m,:);
c = c(m,:);

hold('on');
for n = 1:length(m)
    viscircles(p(n,:), o.geo_int, 'Color', c(n,:));
end
hold('off');

% Scale image
axis('image');
if o.par_dim >= 1; xlim([o.geo_min(1), o.geo_max(1)]); end
if o.par_dim >= 2; ylim([o.geo_min(2), o.geo_max(2)]); end
if o.par_dim >= 3; zlim([o.geo_min(3), o.geo_max(3)]); end

end

