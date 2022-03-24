%
%   This uses hashes to compute energy
%

clear
close all

% Seed the RNG
rng(0)

% Particles to simulate
N=10000;
r = rand(2,N);

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
af = 0.05;              % range and strength of viscous force
f0 = 10;                % 
Nba = 2;                % 
bf = Nba*af;            % Hashing box size
sf = af;                % Stride for scanning
Nb = 1/af;              % Number of boxes per side
Nb2 = Nb*Nb;            % Total number of boxes
S = sparse(Nb2,Nb2);    % sparse matrix of box-box interactions

xb = [1:Nb]*sf-sf/2;    % X values of centers
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


while 1

    %
    %   Figure out which box (hash) a particle belongs to
    %

    indxnew = ceil(r(1,:)/sf);
    indynew = ceil(r(2,:)/sf);
    
    indxnew(find(indxnew==0))=1;
    indynew(find(indynew==0))=1;
    indxnew(find(indxnew>Nb))=Nb;
    indynew(find(indynew>Nb))=Nb;
    
    indbnew = (indxnew-1)*Nb+indynew;       % this is a box number for each particle
    
    dindb = indbnew-indb;                   % difference
    indDiff = find(dindb);
    
    for i=1:length(indDiff)                 % only deal with points that changed their hash
        p = indDiff(i);                     % point in question
        LLb{indb(p)} = setxor(LLb{indb(p)}, p);     % remove the point
        LLb{indbnew(p)} = union(LLb{indbnew(p)}, p);     % add the point at the new place
    end
    
    indb = indbnew;

    % compute force
    % LLb contains the list of points for each box
    
    f = v*0;
    
    for ib = 1:Nb2                         % go through all boxes
        indCurrBox = LLb{ib};              % particles in this box
        listOfBoxes = LoB{ib};
        
        Npts = 0;
        for k=1:length(listOfBoxes)
            kb = listOfBoxes(k);
            np = length(LLb{kb});
            if np>0
                intBuffer((Npts+1):(Npts+np)) = LLb{kb};
                Npts = Npts + np;
            end
        end
        
        indIntPart = intBuffer(1:Npts);
        
        %indIntPart = [LLb{listOfBoxes}]; 
        
        if length(indIntPart)>1
 
        ro = r(:,indIntPart);           % positions of other particles
        vo = v(:,indIntPart);           % velocities of other particles
        ooo = ones(1,size(ro,2));
        
 
        for ip = 1:length(indCurrBox)
            p = indCurrBox(ip);         % index of particle
            rp = r(:,p);
            vp = v(:,p);
        
            dr = rp * ooo - ro;
            absdr = sqrt(sum(dr.^2));
            q = dr ./ ([1 1]'*(absdr+1e-6)); % Force is 1/r
            
            dv = vp * ooo - vo;
            ff = [1,1]' * (absdr < af);    % form-factor for force
            f(:,p) =  sum(f0*ff.*q,2);
        end
        end
    end
    
    %
    %   End of hashing (box calculation) + force
    %
    
    rnew = r+v*dt;
    vnew = v+f*dt;
    
    if 1            % reflective BC
    
        stdv = 0.1;
        
        %
        %   diffusive hitting x=0
        %

        if 0
            
        ind = find(rnew(1,:)<0);
        
        xhit = rnew(1,ind)*0;
        yhit = r(2,ind)+(rnew(2,ind)-r(2,ind)) .* abs(r(1,ind)) ./ abs(rnew(1,ind)-r(1,ind));
        rhit = [xhit; yhit];
        dr = rnew(:,ind)-rhit;               % violation
        absdr = sqrt(sum(dr.^2));
        
        absv = sqrt(sum(v(:,ind).^2));
        vfac = exp(stdv*randn(size(xhit)));
        absvnew = absv .* vfac;
        th = asin(2*rand(size(xhit))-1);    % new angle
        
        qnew = [cos(th);sin(th)];           % normal vector
        vnew(1,ind) = absvnew .* qnew(1,:);
        vnew(2,ind) = absvnew .* qnew(2,:);
        rnew(1,ind) = rhit(1,:) + absdr .* qnew(1,:);
        rnew(2,ind) = rhit(2,:) + absdr .* qnew(2,:);
        
        else    % reflective BC
            
                ind = find(rnew(1,:)<0);
            vnew(1,ind)=-v(1,ind);
            rnew(1,ind) = abs(rnew(1,ind));
        end
        
        %
        %   diffusive hitting y=0
        %
        
        u=10;                            % speed of the wall
        
        ind = find(rnew(2,:)<0);
        
        yhit = rnew(2,ind)*0;
        xhit = r(1,ind)+(rnew(1,ind)-r(1,ind)) .* abs(r(2,ind)) ./ abs(rnew(2,ind)-r(2,ind));
        rhit = [xhit; yhit];
        dr = rnew(:,ind)-rhit;               % violation
        absdr = sqrt(sum(dr.^2));
        
        vp = v(:,ind);
        vp(1,:) = vp(1,:)-u;                % velocity in the sr of moving wall
        
        absvp = sqrt(sum(vp.^2));
        
        vfac = exp(stdv*randn(size(xhit))); % randomization
        absvpnew = absvp .* vfac;
        
        th = asin(2*rand(size(xhit))-1);    % new angle in the sr of the wall
        
        qnewp = [sin(th);cos(th)];           % normal vector
        vnewp = vnew(:,ind);
        vnewp(1,:) = absvpnew .* qnewp(1,:);
        vnewp(2,:) = absvpnew .* qnewp(2,:);
        
        vnewnew = vnewp;
        vnewnew(1,:) = vnewp(1,:)+u;
        
        absvnewnew = sqrt(sum(vnewnew.^2));
        qnew = vnewnew ./ ([1 1]'*(absvnewnew+1e-6));
        absvnew = sqrt(sum(vnew(:,ind).^2));
        rat = absvnewnew ./ (absvnew+1e-6);
        
        rnew(1,ind) = rhit(1,:) + absdr .* qnew(1,:) .* rat;
        rnew(2,ind) = rhit(2,:) + absdr .* qnew(2,:) .* rat;
        
        vnew(:,ind) = vnewnew;
        

        %
        %
        %
        
        ind = find(rnew(1,:)>1);
        vnew(1,ind)=-v(1,ind);
        rnew(1,ind) = rnew(1,ind)-2*abs(rnew(1,ind)-1);
    
        %
        %
        %
        
        ind = find(rnew(2,:)>1);
        vnew(2,ind)=-v(2,ind);
        rnew(2,ind) = rnew(2,ind)-2*abs(rnew(2,ind)-1);
    
    else            % diffusive BCs
        
        ind = find(rnew(1,:)<0);
        
        
        vnew(1,ind)=-v(1,ind);
        rnew(1,ind) = abs(rnew(1,ind));
    end
        
        
        
        
        
    
    
    r = rnew;
    v = vnew;
    v=v/sqrt(mean(var(v)))*Vave;
    
    for i=1:Np_show
        %r10(:,2:end) = r10(:,1:(end-1));
        %r10(:,1)=r(:,10);
        rshow(:,2:end,i) = rshow(:,1:(end-1),i);
        rshow(:,1,i) = r(:,ind_show(i));
    end
    
    if ~mod(step,Disp)
        
        if mod(step*dt,1)==0
            ind_turb_show = LLb{round(Nb2/2)+1};
        end
        
        
        figure(10)
        plot(rnew(1,:), rnew(2,:), '.','color', 0.7*[1 1 1]), axis equal, hold on
        line([0 1 1 0 0], [1 1 0 0 1], 'color', 'r')
        %plot(r10(1,:),r10(2,:), 'r')
        
        for i=1:Np_show
            plot(rshow(1,:,i), rshow(2,:,i), 'color', show_col(:, i))
        end
        
        if exist('ind_turb_show')
            plot(rnew(1,ind_turb_show), rnew(2,ind_turb_show), '.','color', 'r')
        end
        
        title(sprintf('t=%g, T=%g', step*dt, mean(v(:).^2)))
        hold off
        axis([-0.1 1.1 -0.1 1.1])
        drawnow
        
        figure(20)
        
        if ~exist('Vx')
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
        
        subplot(1,2,1)
        imagesc(Vx), axis image, colorbar, colormap jet, 
        set(gca, 'ydir', 'normal'), drawnow
        
        subplot(1,2,2)
        imagesc(Vy), axis image, colorbar, colormap jet, 
        set(gca, 'ydir', 'normal'), drawnow
        

        figure(40)
        
        xb = [1:Nb]-0.5;
        yb = xb;
        [Xb,Yb]= meshgrid(xb,yb);
        Vmax = max(sqrt(Vx(:).^2  + Vy(:).^2));
        Qv = [Vx(:),Vy(:)]';
        Qv = Qv ./ ([1 1]'*sqrt(sum(Qv.^2)));
        %quiver(Xb(:),Yb(:), Vx(:)/Vmax, Vy(:)/Vmax), axis image 
        quiver(Xb(:),Yb(:), Qv(1,:)', Qv(2,:)'), axis image 
        drawnow
        
        
        
        
    end
    
    
    step = step+1;
    pause(0.01)
end
