classdef Particle < handle
    %PARTICLE SPH particles class
    %   Contains info on the particles
    
    properties
        x
        v
        f
        rho
        p
        rad
        lim
        mass
        visc
        grav
        dist
        ehat
        figs
        gasConst
        rhoRefer
    end
    
    methods
        function o = Particle(cor)
            %PARTICLE Construct an instance of this class
            o.x = cor;
            o.v = zeros(size(cor));
            o.f = zeros(size(cor));
            o.rho = zeros(size(cor,1),1);
            o.p = zeros(size(cor,1),1);
            o.grav = [0,-9.8];
        end
        
        function o = ComputeDistance(o)
            % COMPUTEDIST Compute the distance matrix of particles
            dd = zeros(size(o.x,1));
            for ii = 1:size(o.x,2)
                dd = dd + ( (o.x(:,ii).') - (o.x(:,ii)) ).^2; 
            end
            [i,j] = find(dd<(o.rad^2));
            k = sub2ind(size(dd),i,j);
            o.dist = sparse(i,j,dd(k),size(dd,1),size(dd,2));
            o.ehat = cell(size(o.x,2),1);
            for ii = 1:size(o.x,2)
                vv = (o.x(i,ii)-o.x(j,ii))./dd(k);
                o.ehat{ii} = sparse(i,j,vv,size(dd,1),size(dd,2));
            end
        end
        
        function KER = ComputeKernel(o,fun)
            % COMPUTEKERNEL Compute the kernel function
            KER = spfun(@(x) fun(x,o.rad),o.dist) + ...
                fun(0,o.rad)*speye(size(o.dist,1));
        end
        
        function o = ComputeDensity(o)
            % COMPUTEDENSITY Compute the density function
            poly6 = @(r,h) (max((h^2)-(r.^2),0).^3) * ...
                315 / (64*pi*(h^9));
            o.rho = o.mass*full(sum(o.ComputeKernel(poly6),2));
        end
        
        function o = ComputePressure(o)
            % COMPUTEPRESSURE Compute the pressure function
            o.p = o.gasConst*(o.rho-o.rhoRefer);
        end
        
        function o = ComputeForces(o)
            % COMPUTEFORCES Compute the force function
            spiky_der = @(r,h) (max(h-r,0).^2) * (-3) * ...
                15 / (pi*(h^6));
            viskr_lap = @(r,h) max(h-r,0) * 45 / (pi*(h^6));
            % Forces
            f_pre = zeros(size(o.x));
            f_vis = zeros(size(o.x));
            f_grv = ones(size(o.x)) .* o.grav;
            for i = 1:size(o.x,2)
                % Pressure force
                f_pre(:,i) = o.mass * full(sum(...
                    o.ehat{i} .* ( .5*(o.p+(o.p'))./(o.rho')) .* ...
                    o.ComputeKernel(spiky_der) ,2));
                % Viscosity force
                f_vis(:,i) = o.mass * o.visc * full(sum(...
                    (((o.v(:,i)') - o.v(:,i))./(o.rho')) .* ...
                    o.ComputeKernel(viskr_lap) ,2));
            end
            o.f = f_pre + f_vis + f_grv;
        end
        
        function o = Integrate(o,dt)
            % INTEGRATE Move points in simulation
            DAMP = -.5;
            
            o.v = o.v + dt * o.f ./ o.rho;
            o.x = o.x + dt * o.v;
            
            % Enforce boundary
            sto = o.x(:,1) < o.lim(1,1) + o.rad;
            o.v(sto,1) = DAMP * o.v(sto,1);
            o.x(sto,1) = o.lim(1,1) + o.rad;
            
            sto = o.x(:,1) > o.lim(1,2) - o.rad;
            o.v(sto,1) = DAMP * o.v(sto,1);
            o.x(sto,1) = o.lim(1,2) - o.rad;
            
            sto = o.x(:,2) < o.lim(2,1) + o.rad;
            o.v(sto,2) = DAMP * o.v(sto,2);
            o.x(sto,2) = o.lim(2,1) + o.rad;
            
            sto = o.x(:,2) > o.lim(2,2) - o.rad;
            o.v(sto,2) = DAMP * o.v(sto,2);
            o.x(sto,2) = o.lim(2,2) - o.rad;
        end
        
        function RenderFrame(o)
            % RENDERFRAME Show points
            if isempty(o.figs)
                o.figs = figure(5);
            end
            o.figs;
            scatter(o.x(:,1), o.x(:,2), 50, 'filled');
            axis('equal');
            axis([o.lim(1,1), o.lim(1,2), o.lim(2,1), o.lim(2,2)]);
            drawnow;
        end
    end
end