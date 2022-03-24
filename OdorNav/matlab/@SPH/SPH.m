classdef SPH < handle
%SPH This class holds and performs simulation stuff
    properties
        name
        %-----Simulation parameters
        stepSize
        stepNo
        time
        dimension
        support
        gravity
        kernel
        stateEquation
        sound
        corMin
        corMax
        smoothing
        % Referance values
        refSpeed
        refPressure
        refDensity
        refFriction
        %-----Particle quantities
        cor
        vel
        mom
        den
        pre
        mas
        vol
        acc
        fri
        forcePre
        forceVis
        forceAdv
        forceExt
        forceBkg
        inletRegion
        outletRegion
        %-----Interactions
        intMask     % Matrix of interactions
        intLabel    % List of interactions
        kerVal      % Kernel values
        kerDer      % Kernel derivative values
        distance    % Distance between particles
        velocity    % Velocity difference between particles
        distHat     % distance hat vector
        velHat      % velocity difference hat vector
        %-----To hide later
        f_fluid     % Particle is a fluid particle
        f_inlet     % Particle is in inflow region
        f_outlet    % Particle in in outflow region
        f_bound     % Particle is a boundary particle
        % Functions
        fn_ker      % Functional for kernel
        fn_kdr      % Functional for kernel derivative
        fn_pre      % Function to get pressure from density
        fn_den      % Function to get density from pressure
        % Hashing stuff
        parVox      % Particle's voxel
        voxPar      % Boolean matrix of particle within voxel
        voxLst      % List of particles within voxel
        voxLen      % Total number of particles in voxel
        parAdj      % Particle adjacency matrix (this is the hard calculate)
        parOff      % Particle offset value (used to calculate hash)
        voxOff      % Voxel offset value (used to calculate hash)
        voxDim      % Number of voxels in each dimension
        voxAdj      % Voxel adjacency mapping
        % Handles for figures
        fig_momenta
        fig_interactions
        fig_particles
        fig_forces
    end
    properties (Dependent)
        particleNo
    end
    properties (Dependent) % Make hidden
        fluidNo
        boundNo
        inletNo
        outletNo
        f_none
        noneNo
        voxNum      % Number of voxels
    end
    
    methods
        function o = SPH()
            %PARTICLES Construct an instance of this class
        end
        
        function v = get.particleNo(o)
            v = size(o.cor, 1);
        end
        
        function v = get.fluidNo(o)
            v = sum(o.f_fluid);
        end
        
        function v = get.boundNo(o)
            v = sum(o.f_bound);
        end
        
        function v = get.inletNo(o)
            v = sum(o.f_inlet);
        end
        
        function v = get.outletNo(o)
            v = sum(o.f_outlet);
        end
        
        function v = get.f_none(o)
            v = ~(o.f_bound | o.f_fluid | o.f_inlet | o.f_outlet);
        end
        
        function v = get.noneNo(o)
            v = size(o.cor, 1) - sum( ...
                o.f_bound | o.f_fluid | o.f_inlet | o.f_outlet);
        end
        
        function v = get.voxNum(o)
            v = prod(o.voxSize);
        end
        
    end
end

