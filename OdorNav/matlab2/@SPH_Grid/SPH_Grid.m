classdef SPH_Grid < SPH
    %SPH_VOXEL SPH equations, solved with sub-gridding
    properties
        grid_sub
        grid_len
        grid_dim
        grid_mas
        grid_den
        grid_vel
        grid_pre
        grid_pre_grad
        grid_for_pre
        grid_for_vis
        filt_ker
        filt_ker_grad
        filt_ker_grad_ten
        filt_rad
        pre_kgt
    end
    methods
        function o = SPH_Grid(N)
            %SPH_TV Construct an instance of this class
            o@SPH(N);
        end
        calculateQuantities(o)
        calculateMovement(o)
    end
end

