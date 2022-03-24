classdef SPH < handle
%SPH This class holds and performs simulation stuff
    properties
        %-----Simulation info
        sim_name
        sim_step
        sim_iter
        sim_state
        sim_hooks
        %-----Parameters
        par_sep
        par_snd
        par_spd
        par_pre
        par_den
        par_vis
        par_grv
        par_mas
        par_vol
        %-----Geometry
        geo_min
        geo_max
        geo_inl_min
        geo_inl_max
        geo_out_min
        geo_out_max
        geo_smt
        geo_int
        geo_ker
        %-----Particles
        prt_pos
        prt_vol
        prt_vel
        prt_mom
        prt_den
        prt_pre
        prt_mas
        prt_acc
        prt_fri
        prt_for_pre
        prt_for_vis
        prt_for_adv
        prt_for_ext
        prt_for_cor
        prt_fld
        prt_bdr
        prt_inl
        prt_out
        %-----Calculated quantities
        cal_dist
        cal_ehat
        cal_kern
        cal_kgrd
        cal_int_i
        cal_int_j
        cal_int_ij
        %-----Internal
        int_kf
        int_kd
        int_pre
        int_den
        %-----Hashing
        hsh_prt_adj         % Particle adjacency matrix
        hsh_prt_vox         % Particle's voxel
        hsh_prt_loc         % Location of particle ID in voxLst
        hsh_vox_adj         % Voxel adjacency <LIST>
        hsh_vox_adj_mat     % Voxel adjacency matrix
        hsh_vox_prt         % Voxel to particles <LIST>
        hsh_vox_len         % Total number of particles in voxel
        hsh_offset          % Voxel offset value (used to calculate hash)
        hsh_range           % Number of voxels in each dimension
        %-----Figure handles
        fig_mom
        fig_int
        fig_prt
        fig_for
    end
    properties (Dependent)
        sim_time
        par_dim
        prt_for
        prt_num
        prt_sim
        prt_nsm
        prt_pos_fld
        prt_pos_bdr
        prt_pos_inl
        prt_pos_out
        prt_fld_num
        prt_bdr_num
        prt_inl_num
        prt_out_num
        prt_sim_num
        prt_nsm_num
        prt_fld_id
        prt_bdr_id
        prt_inl_id
        prt_out_id
        prt_sim_id
        prt_nsm_id
        hsh_prt_adj_i
        hsh_prt_adj_j
        hsh_vox_num
    end
    
    methods
        %-----INIT
        function o = SPH(N)
            %PARTICLES Construct an instance of this class
            o.sim_iter = 0;
            o.prt_pos = [];
            if exist('N','var')
                o.sim_name = N;
            else
                o.sim_name = '';
            end
            o.hsh_vox_adj = []; % Leave empty so can be filled up later.
            o.hsh_prt_adj = [];
            % Initialize as bool
            o.prt_fld = false(0, 1);
            o.prt_bdr = false(0, 1);
            o.prt_inl = false(0, 1);
            o.prt_out = false(0, 1);
            % No hooks for now
            o.sim_hooks = {};
        end
    end
    methods
        %-----DEPENDENT
        function v = get.sim_time(o)
            v = o.sim_step * o.sim_iter;
        end
        function v = get.par_dim(o)
            v = size(o.prt_pos,2);
        end
        function v = get.prt_for(o)
            v = o.prt_for_pre + o.prt_for_vis + o.prt_for_adv + o.prt_for_adv;
        end
        function v = get.prt_num(o)
            v = size(o.prt_pos, 1);
        end
        function v = get.prt_sim(o)
            v = (o.prt_fld | o.prt_bdr | o.prt_inl | o.prt_out);
        end
        function v = get.prt_nsm(o)
            v = ~(o.prt_fld | o.prt_bdr | o.prt_inl | o.prt_out);
        end
        function v = get.prt_fld_num(o)
            v = sum(o.prt_fld);
        end
        function v = get.prt_bdr_num(o)
            v = sum(o.prt_bdr);
        end
        function v = get.prt_inl_num(o)
            v = sum(o.prt_inl);
        end
        function v = get.prt_out_num(o)
            v = sum(o.prt_out);
        end
        function v = get.prt_pos_fld(o)
            v = o.prt_pos(o.prt_fld,:);
        end
        function v = get.prt_pos_bdr(o)
            v = o.prt_pos(o.prt_bdr,:);
        end
        function v = get.prt_pos_inl(o)
            v = o.prt_pos(o.prt_inl,:);
        end
        function v = get.prt_pos_out(o)
            v = o.prt_pos(o.prt_out,:);
        end
        function v = get.prt_sim_num(o)
            v = sum(o.prt_fld | o.prt_bdr | o.prt_inl | o.prt_out);
        end
        function v = get.prt_nsm_num(o)
            v = size(o.prt_cor, 1) - ...
                sum(o.prt_fld | o.prt_bdr | o.prt_inl | o.prt_out);
        end
        function v = get.prt_fld_id(o)
            v = find(o.prt_fld);
        end
        function v = get.prt_bdr_id(o)
            v = find(o.prt_bdr);
        end
        function v = get.prt_inl_id(o)
            v = find(o.prt_inl);
        end
        function v = get.prt_out_id(o)
            v = find(o.prt_out);
        end
        function v = get.prt_sim_id(o)
            v = find(o.prt_fld | o.prt_bdr | o.prt_inl | o.prt_out);
        end
        function v = get.prt_nsm_id(o)
            v = find(~(o.prt_fld | o.prt_bdr | o.prt_inl | o.prt_out));
        end
        function v = get.hsh_prt_adj_i(o)
            [v,~] = find(o.hsh_prt_adj);
        end
        function v = get.hsh_prt_adj_j(o)
            [~,v] = find(o.hsh_prt_adj);
        end
        function v = get.hsh_vox_num(o)
            v = prod(o.hsh_range);
        end
    end
    methods(Sealed=false)
        calculateDistance(o)
        calculateHash(o)
        chooseKernel(o, KER)
        chooseStateEqn(o, EQN, varargin)
        plotGrid(o)
        plotGridCircles(o, want)
        plotInteraction(o, cl1, cl2, adj)
        plotParticles(o)
        p = plotVectors(o, prt, vec, scl)
        showParticles(o)
        showVelocities(o, want)
        showForces(o, want)
    end
    methods(Abstract=true)
        calculateQuantities(o)
        calculateMovement(o)
    end
    methods(Static=true, Sealed=true)
        out = getColors(N, sat)
        out = getAdjacent(D)
        out = getS2I(shp, sub)
        out = getI2S(shp, ind)
        out = getFocusedFig(out)
        [I, R, K] = getPartitions(A,M)
        [I, X] = getAbsPartitions(A)
    end
end


