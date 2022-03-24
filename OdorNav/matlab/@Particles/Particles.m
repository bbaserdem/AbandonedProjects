classdef Particles < handle
    %PARTICLES Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        r       % Positions
        v       % Advection velocity
        u       % Momentum velocity
        d       % Density of particle
        p       % Pressure of particle
        m       % Mass of particle
        s       % Volume of particle
        f       % Force on particle
        a       % Particle acceleration
        n       % Kinetic friction coefficient
        %-----HASHING
        subHash % Hash as subscripts
        linHash % Hash as linear index
        parHash % Particles with specific hash index
        parLen  % Particles per hash
        adjHash % List of particles that are adjacent to this
        adjLen  % Particles per adjacency
        adjRHash% List of particles that are reverse adjacent to this
        adjRLen % Particles per reverse adjacency
    end
    properties (Dependent)
        dim             % Problem dimensionality
        num             % Particle number
    end
    
    methods
        function o = Particles(cord, dens, volm, kinv)
            %PARTICLES Construct an instance of this class
            if ~exist('dens', 'var')
                dens = 1;
            end
            if ~exist('volm', 'var')
                volm = 1;
            end
            if ~exist('kinv', 'var')
                kinv = 1;
            end
            o.r = cord;
            o.v = zeros(size(o.r));
            o.u = zeros(size(o.r));
            o.d = dens .* ones(length(o.r), 1);
            o.p = zeros(length(o.r), 1);
            o.s = volm .* ones(length(o.r), 1);
            o.m = o.s .* o.d;
            o.f = zeros(size(o.r));
            o.n = kinv .* ones(length(o.r), 1);
            o.a = zeros(size(o.r));
        end
        
        function v = get.dim(o)
            v = size(o.r, 2);
        end
        function v = get.num(o)
            v = size(o.r, 1);
        end
    end
end

