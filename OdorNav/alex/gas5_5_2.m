%
%   This uses a potential field to compute force

clear
close all

% Seed the RNG
rng(0)

% Particles to simulate
N=10000;
r = rand(2,N);

% Wall velocity
wall_u = 10;
stdv = .1;

% Particles to track trajectories of
Np_show = 10;
ind_show = randperm(N);
ind_show = ind_show(1:Np_show);

% Average velocity
Vave = 3;
v = Vave*randn(2,N);

% Time stepping
dt = 0.001;
step = 0;

% Flag to choose to display
Disp = 1;

% Force constants
af = 0.05;              % Interaction range
f0 = 5;                 % 
Nba = 2;                % 
bf = Nba*af;            % Hashing box size
sf = af;                % Stride for scanning
Nb = 1/af;              % Number of boxes per side
Nb2 = Nb*Nb;            % Total number of boxes
S = sparse(Nb2,Nb2);    % sparse matrix of box-box interactions

% Mean field generation
sg_n = 10;                   % Subgrid resolution wrt interaction range
sg_d = af / sg_n;           % subgrid dx
sg_s = 1 / sg_d;            % subgrid spaces
sg_f = 2 * sg_n + 1;        % subgrid filter size
sto = linspace(-af, af, sg_f);
[sg_fy, sg_fx] = meshgrid(sto, sto);
% Calculate filter distance and hat
filt_dist = sqrt( (sg_fy.^2) + (sg_fx.^2) );
filt_haty = sg_fy ./ (filt_dist + 1e-10);
filt_hatx = sg_fx ./ (filt_dist + 1e-10);


xb = (1:Nb)*sf-sf/2;    % X values of centers
yb = xb;                % Y values of centers

% Find interactions
[Xb,Yb]=meshgrid(xb, yb);

% File name for storage
fname = sprintf('box_interaction_matrix%ux%u.mat', Nb, Nb);
d = dir(fname);

% Find which box is adjacent to which
if ~isempty(d)
    load(fname,'S');
    disp('Loaded form file')
else
    for ix = 1:Nb
        for iy = 1:Nb
            i = iy+Nb*(ix-1);
            for jx = (ix-Nba):(ix+Nba)
                jx = min(Nb,max(jx,1));
                for jy = (iy-Nba):(iy+Nba)
                    jy = min(Nb,max(jy,1));
                    j = jy+Nb*(jx-1);
                    S(i,j) = ((Xb(i)-Xb(j))^2 + (Yb(i)-Yb(j))^2 < bf^2);
                end
            end
        end
    end
    save(fname)
end
% list of boxes interacting with this one
LoB = cell(Nb2,1);
for i=1:Nb2
    LoB{i} = find(S(i,:));
end

%   Figure out which box (hash) a particle belongs to
%

indx = ceil(r(1,:)/sf);
indy = ceil(r(2,:)/sf);
    
indb = (indx-1)*Nb+indy;                % this is a box number for each particle
[uB, Lb, Ib, Jb] = unique_list(indb');   % Lb is the list of points in boxes
    
for i=1:length(Lb)
    Lb{i} = Lb{i}';
end

LLb = cell(1, Nb2);                      % LLb will contain lists of points in each box
for i=1:length(uB)
   LLb{uB(i)} = Lb{i};
end

intBuffer = zeros(N,1);
Npts = 0;

%r10 = r(:,10)*ones(1,1000);
rshow = zeros(2,200,Np_show);
show_col = zeros(3,Np_show);
for i=1:Np_show
   rshow(:,:,i) = r(:,ind_show(i))*ones(1,size(rshow,2)); 
   show_col(:,i) = rand(3,1);
end

FIG = figure(37);

%-----FORCES
force_profile = 'gaussian';
switch force_profile
    case 'flat'
        % Constant repulsive force within interaction radius
        FORCE = @(r) ones(size(r));
        POTEN = @(r) max(0, af - r);
    case 'gaussian'
        % Force profile: normal distribution
        POTEN = @(r) .5*sqrt(pi)*erf(r/(sqrt(2)*af/2));
        FORCE = @(r) normpdf(r, 0, af/2);
    case 'inverse squared'
        % Force profile: inverse square
        POTEN = @(r) -(r.^-1);
        FORCE = @(r) r.^-2;
end
filt_x = - filt_hatx .* (filt_dist <= af) .* FORCE(filt_dist);
filt_y = - filt_haty .* (filt_dist <= af) .* FORCE(filt_dist);

while 1

    %-----------------%
    %-----HASHING-----%
    %-----------------%

    indxnew = ceil(r(1,:)/sf);
    indynew = ceil(r(2,:)/sf);
    
    indxnew(indxnew==0)=1;
    indynew(indynew==0)=1;
    indxnew(indxnew>Nb)=Nb;
    indynew(indynew>Nb)=Nb;
    
    indbnew = (indxnew-1)*Nb+indynew;       % this is a box number for each particle
    
    dindb = indbnew-indb;                   % difference
    indDiff = find(dindb);
    
    for i=1:length(indDiff)                 % only deal with points that changed their hash
        p = indDiff(i);                     % point in question
        LLb{indb(p)} = setxor(LLb{indb(p)}, p);     % remove the point
        LLb{indbnew(p)} = union(LLb{indbnew(p)}, p);     % add the point at the new place
    end
    
    indb = indbnew;
    
    %---------------%
    %-----FORCE-----%
    %---------------%
    
    % Get particle subhash, and counts per voxel
    parhash = ceil(r/sg_d)';
    [shash, shash_count] = alex_unique_abu(parhash);
    meanField_c = full(sparse(shash(:,1), shash(:,2), shash_count, sg_s, sg_s));
    
    % Filter the counts per voxel image to get the forces
    meanField_x = imfilter(meanField_c, filt_x, 'same');
    meanField_y = imfilter(meanField_c, filt_y, 'same');
    
    % Reclaim force on particle
    parhash_lin_i = 1 + (parhash(:,1)-1) + sg_s.*(parhash(:,2)-1);
    f = f0 * ([meanField_x(parhash_lin_i), meanField_y(parhash_lin_i)]');
    
    %------------------%
    %-----MOVEMENT-----%
    %------------------%
    
    % Move particles
    rnew = r+v*dt;
    vnew = v+f*dt;
    
    %-------------------%
    %-----DIFFUSION-----%
    %-------------------%
    
    ind = find(rnew(2,:)<0);
    
    yhit = rnew(2,ind)*0;
    xhit = r(1,ind)+(rnew(1,ind)-r(1,ind)) .* abs(r(2,ind)) ./ abs(rnew(2,ind)-r(2,ind));
    rhit = [xhit; yhit];
    dr = rnew(:,ind)-rhit;               % violation
    absdr = sqrt(sum(dr.^2));
    
    vp = v(:,ind);
    vp(1,:) = vp(1,:)-wall_u;                % velocity in the sr of moving wall
    
    absvp = sqrt(sum(vp.^2));
    
    vfac = exp(stdv*randn(size(xhit))); % randomization
    absvpnew = absvp .* vfac;
    
    th = asin(2*rand(size(xhit))-1);    % new angle in the sr of the wall
    
    qnewp = [sin(th);cos(th)];           % normal vector
    vnewp = vnew(:,ind);
    vnewp(1,:) = absvpnew .* qnewp(1,:);
    vnewp(2,:) = absvpnew .* qnewp(2,:);
    
    vnewnew = vnewp;
    vnewnew(1,:) = vnewp(1,:)+wall_u;
    
    absvnewnew = sqrt(sum(vnewnew.^2));
    qnew = vnewnew ./ ([1 1]'*(absvnewnew+1e-6));
    absvnew = sqrt(sum(vnew(:,ind).^2));
    rat = absvnewnew ./ (absvnew+1e-6);
    
    rnew(1,ind) = rhit(1,:) + absdr .* qnew(1,:) .* rat;
    rnew(2,ind) = rhit(2,:) + absdr .* qnew(2,:) .* rat;
    
    vnew(:,ind) = vnewnew;
    
    %--------------------%
    %-----REFLECTION-----%
    %--------------------%
    
    % x = 0
    ind = find(rnew(1,:)<0);
    vnew(1,ind)=-v(1,ind);
    rnew(1,ind) = abs(rnew(1,ind));
    
    % x = 1
    ind = find(rnew(1,:)>1);
    vnew(1,ind)=-v(1,ind);
    rnew(1,ind) = rnew(1,ind)-2*abs(rnew(1,ind)-1);
    
    % y = 1
    ind = find(rnew(2,:)>1);
    vnew(2,ind)=-v(2,ind);
    rnew(2,ind) = rnew(2,ind)-2*abs(rnew(2,ind)-1);
        
    % Do the correction
    r = rnew;
    v = vnew;
    v = v/sqrt(mean(var(v)))*Vave;
    
    %------------------%
    %-----PLOTTING-----%
    %------------------%
    
    for i=1:Np_show
        rshow(:,2:end,i) = rshow(:,1:(end-1),i);
        rshow(:,1,i) = r(:,ind_show(i));
    end
    
    if ~mod(step,Disp)
        
        if mod(step*dt,1)==0
            ind_turb_show = LLb{round(Nb2/2)+1};
        end
        set(0, 'CurrentFigure', FIG);
        
        subplot(2,2,1)
        plot(rnew(1,:), rnew(2,:), '.','color', 0.7*[1 1 1]);
        axis('equal');
        hold('on');
        line([0 1 1 0 0], [1 1 0 0 1], 'color', 'r')
        %plot(r10(1,:),r10(2,:), 'r')
        
        for i=1:Np_show
            plot(rshow(1,:,i), rshow(2,:,i), 'color', show_col(:, i))
        end
        
        if exist('ind_turb_show','var')
            plot(rnew(1,ind_turb_show), rnew(2,ind_turb_show), '.','color', 'r')
        end
        
        title(sprintf('t=%g, T=%g', step*dt, mean(v(:).^2)));
        hold('off');
        axis([-0.1 1.1 -0.1 1.1]);
        
        if ~exist('Vx','var')
            Vx = zeros(Nb,Nb);
            Vy = zeros(Nb,Nb);
        end
        
        eee = 0.01;
        for i=1:Nb2
            mvx = mean(v(1,LLb{i}));
            mvy = mean(v(2,LLb{i}));
           if isnan(mvx), mvx = 0; end
           if isnan(mvy), mvy = 0; end
           
           Vx(i) = (1-eee)*Vx(i) + eee*mvx; 
           Vy(i) = (1-eee)*Vy(i) + eee*mvy; 
        end
        
        subplot(2,2,2)
        imagesc(Vx);
        axis('image');
        colorbar;
        colormap('jet'); 
        set(gca, 'ydir', 'normal')
        
        subplot(2,2,4)
        imagesc(Vy);
        axis('image');
        colorbar;
        colormap('jet'); 
        set(gca, 'ydir', 'normal')
        

        subplot(2,2,3);
        
        xb = (1:Nb)-0.5;
        yb = xb;
        [Xb,Yb]= meshgrid(xb,yb);
        Vmax = max(sqrt(Vx(:).^2  + Vy(:).^2));
        Qv = [Vx(:),Vy(:)]';
        Qv = Qv ./ ([1 1]'*sqrt(sum(Qv.^2)));
        %quiver(Xb(:),Yb(:), Vx(:)/Vmax, Vy(:)/Vmax), axis image 
        quiver(Xb(:),Yb(:), Qv(1,:)', Qv(2,:)');
        axis('image');
        
    end
    
    
    step = step+1;
    drawnow;
end
