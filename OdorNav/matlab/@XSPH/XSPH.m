classdef XSPH < handle
    %PARTICLES Class that contains particles and properties of particles
    %   This class holds information on all the particles; and contains
    %       functions that calculate quantities.
    
    properties
        name
        %-----PHYSICAL QUANTITIES
        fluid
        bound
        inlet
        outlet
        %-----INTERACTIONS (Sparse matrices, or cells of it)
        % Adjacency info
        F_adj           % Voxel neighbours (fluid to fluid)
        B_adj           % Voxel neighbours (fluid to border)
        % Fluid to fluid
        Fi_ij           % Index of interaction; i to j
        Fj_ij           % Index of interaction; i to j
        Fr_ij           % Distance
        Fw_ij           % Kernel
        Fg_ij           % Kernel gradient
        Fe_ij           % Hat vectors
        % Fluid to border
        Bi_ij           % Index of interaction; i to j
        Bj_ij           % Index of interaction; i to j
        Br_ij           % Distance
        Bw_ij           % Kernel
        Bg_ij           % Kernel gradient
        Be_ij           % Hat vectors
        %-----FORCES
        F_pre           % Force due to pressure terms
        F_adv           % Diffusion due to momentum not matching advection
        F_vis           % Force due to viscous forces
        F_ext           % External forces
        F_bkg           % Background pressure correction force
        %-----PARAMETERS
        h               % Smoothing radius
        s               % Support radius, units in smoothing radius
        g               % Gravity
        dt              % Time-step
        T               % Time
        f               % frame-number
        u0              % Referance velocity
        c0              % Speed of sound
        p0              % Referance pressure
        pb              % Background pressure
        d0              % Referance density
        locMax          % Limits for location
        locMin
        %-----HASHING
        hashOffset      % Offset of hash subscript id's
        hashSubDim      % Subscript dim for hash
        hashTicks       % Tick values for hashing
        hashAdj         % Adjacent hashes
        hashAdjRev      % Reverse adjacency
        hashBoundCand   % Boundary candidates for this hash
        hashBoundCandLen
    end
    properties (Dependent)
        dim             % Problem dimensionality
        fNum            % Particle number
        bNum            % Boundary particle number
        sup             % Support in units
        hashLinDim      % Linear dimension for hash
    end
    properties (Hidden)
        figHandles
        %-----MEMOIZATION
        memGetIJ
        memColors
        %-----FUNCTIONS (functions of some r and r0)
        kernel          % Smoothing kernel
        kerDer          % Smoothing kernel derivative
        pressure        % Pressure equation from density
        density         % Density equation from pressure
        %-----Figure handles
        figMom
        figInt
        figPar
        figFor
    end
    
    methods
        function o = XSPH(tit,fld,bnd)
            %PARTICLES Construct an instance of this class
            if exist('tit','var')
                o.name = tit;
            else
                o.name = ['XSPH-sim: ',datestr(now,'dd/mm/yyyy_HH:MM:SS')];
            end
            % Start figure handles storage
            o.figHandles = struct;
            % Memoize functions
            o.memGetIJ = memoize(@aux.get_ij);
            o.memGetIJ.CacheSize = 2000;
            o.memColors = memoize(@(u) aux.get_rainbow(u, .75));
            % Insert data
            if exist('fld','var')
                o.fluid = fld;
            end
            if exist('bnd','var')
                o.bound = bnd;
            end
            o.T = 0;
            o.f = 1;
        end
        
        function v = get.dim(o)
            v = o.fluid.dim;
        end
        function v = get.fNum(o)
            v = o.fluid.num;
        end
        function v = get.bNum(o)
            v = o.bound.num;
        end
        function v = get.sup(o)
            v = o.s * o.h;
        end
        function v = get.hashLinDim(o)
            v = prod(o.hashSubDim);
        end
    end
end

