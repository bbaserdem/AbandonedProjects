classdef loc_SPH < handle
    %LOC_SPH Summary of this class goes here
    %   Detailed explanation goes here
    
    properties%( SetAccess = protected )
        thr         % Threshold radius
        dim         % Dimensionality of the geometry
        phySize     % The physical dimensions of system
        voxSize     % Number of voxels per dimension
        voxPts      % Points within voxel
        ptsCor      % Point actual coordinates
        ptsLoc      % Points location
        ptsVox      % Points in which voxel
        ptsNum      % Points number
        conList     % List of which particle interacts with which
        conDist     % List of connection distances
        conRHat     % Connection hat vectors
    end
    properties%( Access = protected )
        voxNum      % Voxel count; prod(voxSize)
        voxLink     % Which voxel neighbours which (in linear index)
        voxPtsNum   % Number of points within voxel
        voxL2S      % Function to convert linear index to subscript
        voxS2L      % The reverse function
        voxLoc      % Location on the voxels
        disPos      % Positive displacement vectors
        disNum      % Number of positive displacement vectors
    end

    methods
        function o = loc_SPH(rad,geo,poi)
            %LOC_SPH Construct an instance of this class

            % INPUT CHECKING
            % Rotate geo if not horizontal
            if size(geo,1) ~= 1
                geo = geo';
            end
            % Check if radius makes sense
            if ( ~isscalar(rad) ) || ( ~isreal(rad) ) || ( rad(1) <= 0 )
                error('Radius needs to be a >0 real scalar');
            end
            % Check if geometry is divisible;
            if ~all( mod(geo,rad) == 0 )
                error( 'The geometry must be divisible with grid size!' );
            end
            
            % Data entry
            o.thr = rad;
            o.phySize = geo;
            o.dim = length( geo );
            o.voxSize = geo / rad;
            o.voxNum = prod( o.voxSize );

            % Write down converters
            % Do note that the converter function require column input
            o.voxS2L = @(u) 1+sum((u-1).*[1,cumprod(o.voxSize(1:(end-1)))],2);
            % Linear to subscript is a matrix
            o.voxL2S = cell(1,o.dim);
            [o.voxL2S{:}] = ind2sub( o.voxSize , (1:o.voxNum)' );
            o.voxL2S = cell2mat( o.voxL2S );

            % Calculate positive vectors
            o.disNum = ( 3^o.dim - 1 ) / 2;
            % Initiate for N = 1
            o.disPos = 1;
            sto = (-1:1)';
            for d = 2:o.dim
                o.disPos = [ o.disPos, zeros( size( o.disPos, 1 ), 1 ); ...
                    sto, ones( size( sto, 1 ), 1) ];
                sto = [ kron( ones( 3, 1 ), sto ), ...
                    kron( (-1:1)', ones( size( sto, 1 ), 1 ) ) ];
            end

            % Get movement matrix
            o.voxLink = zeros( o.voxNum, o.disNum );
            for l = 1:o.disNum
                o.voxLink(:,l) = o.voxS2L( ...
                    1 + mod( o.voxL2S + o.disPos(l,:) - 1 , o.voxSize ) );
            end
            
            if exist('poi','var')
                o.insertPoints(poi);
            end
        end
        
        function insertPoints(o,pts)
            %INSERTPOINTS: Insert points into the instance
            
            % Input check
            if ( length(size(pts)) ~= 2 ) || ( ~any( size(pts) == o.dim ) )
                error( 'Invalid points dimension' );
            end
            if size(pts,2) ~= o.dim
                pts = pts';
            end
            if any( max(pts,[],1) >= (o.phySize) ) || any( min(pts,[],1) < 0 )
                warning('Points out of bound, wrapping around . . .');
                pts = mod( pts, o.phySize );
            end

            % Insert data
            o.ptsCor = pts;
            o.ptsLoc = mod( pts, o.thr );
            o.ptsNum = size(pts,1);
            o.ptsVox = o.voxS2L( 1 + floor( pts / o.thr ) );

            % Store voxel data
            o.voxPts = cell( o.voxNum, 1 );
            o.voxLoc = cell( o.voxNum, 1 );
            o.voxPtsNum = zeros( o.voxNum, 1 );
            for v = 1:o.voxNum
                o.voxPts{v} = find( o.ptsVox == v )';
                o.voxLoc{v} = o.ptsLoc( o.voxPts{v}, : );
                o.voxPtsNum(v) = length( o.voxPts{v} );
            end
        end
        
        function movePoints(o,del)
            %MOVEPOINTS: Move the points by the specified displacement vector
            o.ptsLoc = o.ptsLoc + del;
            
            % FILL IN LATER
        end
        
        function D = pairwiseDist(o)
            %PAIRWISEDIST Return pairwise distances under length threshold
            % Allocate distance matrix
            allocPts = sum( o.voxPtsNum .^ 2 + ...
                sum( o.voxPtsNum .* o.voxPtsNum(o.voxLink) , 2 ) );
            D2 = sparse( [], [], [], o.ptsNum, o.ptsNum, allocPts );
            rH = cell( 1, o.dim );
            for d = 1:o.dim
                rH{d} = sparse( [], [], [], o.ptsNum, o.ptsNum, allocPts );
            end
            
            % Get square sum of all the coordinate points
            squ = dot( o.ptsLoc , o.ptsLoc , 2 );
            % Get square sum of the displacement vectors
            dis = dot( o.disPos , o.disPos , 2 ) * ( o.thr .^ 2 );
            
            for v = 1:o.voxNum
                % Get the points within this voxel
                pt1 = o.voxPts{v};
                % Get dispacement coordinates within this voxel
                co1 = o.ptsLoc(pt1,:);
                % Get square sum of displacement in voxel
                sq1 = squ(pt1);
                % Do points within the voxel.
                D2(pt1,pt1) = triu( ...
                    sq1 + sq1' - 2 * ( co1 * (co1') ) , 1 );
                for d = 1:o.dim
                    rH{d}(pt1,pt1) = triu( co1(:,d)' - co1(:,d) , 1 );
                end
                % Do crosstalk between voxels
                for l = 1:o.disNum
                    % Get linker vector
                    lin = o.disPos(l,:) * o.thr;
                    % Get the points within adjacent voxel
                    pt2 = o.voxPts{o.voxLink(v,l)};
                    % Get dispacement coordinates within adjacent voxel
                    co2 = o.ptsLoc(pt2,:);
                    % Get square sum of displacement in voxel
                    sq2 = squ(pt2);
                    % Do crosstalk points
                    D2(pt1,pt2) = sq1 + sq2' + dis(l) - 2 * ( ...
                        co1 * (co2') + co1 * (lin') - lin * (co2') );
                    for d = 1:o.dim
                        rH{d}(pt1,pt2) = co2(:,d)' + lin(d) - co1(:,d);
                    end
                end
            end
            % Remove out-of-threshold points
            out = ( D2 >= (o.thr.^2) );
            for d = 1:o.dim
                % Knock out too long vectors
                rH{d}( out ) = 0;
                % Order hat vectors, fliping signs when moving them around
                rH{d} = triu( triu( rH{d}, 1 ) - tril( rH{d}, -1 )', 1 );
            end
            % Remove long lengths
            D2( out ) = 0;
            % Rescale to linear distance
            D2 = sqrt( D2 );
            % Order vectors to { (i,j) | i < j }
            D = triu( D2, 1 ) + tril( D2, -1 )';
            
            % Write results to struct
            o.conList = zeros( nnz(D), 2 );
            [o.conList(:,1),o.conList(:,2),o.conDist] = find( D );
            o.conRHat = zeros( nnz(D), o.dim );
            for d = 1:o.dim
                o.conRHat(:,d) = nonzeros(rH{d}) ./ o.conDist;
            end
            
        end
        
        function D = pairwiseDistBrute(o)
            %PAIRWISEDIST Calculate all to all distance with thresholding
            squ = dot( o.ptsCor , o.ptsCor , 2 );
            D = triu( squ + squ' - 2 * ( o.ptsCor * (o.ptsCor') ) , 1 );
            D( D >= (o.thr.^2) ) = 0;
            D = sqrt(sparse(D));
            
            % Write results to struct
            o.conList = zeros( nnz(D), 2 );
            [o.conList(:,1),o.conList(:,2),o.conDist] = find( D );
            o.conRHat = o.ptsCor(o.conList(:,2),:) - o.ptsCor(o.conList(:,1),:);
            o.conRHat = o.conRHat ./ o.conDist;
        end
        
        function plot(o,brute)
            %PLOT Plot points and the detected connections
            if exist('brute','var')
                if brute
                    o.pairwiseDistBrute
                else
                    o.pairwiseDist
                end
            end
            
            % Check dimensions for 2d
            if o.dim ~= 2
                error('Plotting done only in 2D');
            end
            
            % Get points and ghost points
            col = hsv2rgb( [ linspace(0,1,o.ptsNum)', ones(o.ptsNum,2) ] );
            
            gho_cor = zeros(0,o.dim);
            gho_col = zeros(0,3);
            for d = 1:o.dim
                gho1 = o.ptsCor(:,d) < o.thr;
                if isempty(gho_cor)
                    gho2 = [];
                else
                    gho2 = gho_cor(:,d) < o.thr;
                end
                add = [ o.ptsCor(gho1,:); gho_cor(gho2,:) ];
                addCol = [ col(gho1,:); gho_col(gho2,:) ];
                add(:,d) = add(:,d) + o.phySize(d);
                gho_cor = [gho_cor; add];
                gho_col = [ gho_col; addCol ];
                gho1 = o.ptsCor(:,d) >= o.phySize(d) - o.thr;
                if isempty(gho_cor)
                    gho2 = [];
                else
                    gho2 = gho_cor(:,d) >= o.phySize(d) - o.thr;
                end
                add = [ o.ptsCor(gho1,:); gho_cor(gho2,:) ];
                add(:,d) = add(:,d) - o.phySize(d);
                addCol = [ col(gho1,:); gho_col(gho2,:) ];
                gho_cor = [gho_cor; add];
                gho_col = [ gho_col; addCol ];
            end
            
            % Get plotting coordinates
            x1 = o.ptsCor( o.conList(:,1), 1 )';
            y1 = o.ptsCor( o.conList(:,1), 2 )';
            x2 = x1 + (o.conDist') .* (o.conRHat(:,1)');
            y2 = y1 + (o.conDist') .* (o.conRHat(:,2)');
            x = [x1;x2];
            y = [y1;y2];
            
            p_x = o.ptsCor(:,1)';
            p_y = o.ptsCor(:,2)';
            g_x = gho_cor(:,1)';
            g_y = gho_cor(:,2)';
            
            % Plot the thing
            scatter( p_x, p_y, 20, col, 'filled' );
            hold on
            scatter( g_x, g_y, 20, gho_col );
            
            % Draw lines
            plot(x,y,'Color',[0,0,0,.3]);
            
            % Axis stuff
            axis([-o.thr,o.phySize(1)+o.thr,-o.thr,o.phySize(2)+o.thr]);
            axis image;
            ax = gca;
            ax.XTick = o.thr * ( -1:(o.voxSize(1)+1) );
            ax.YTick = o.thr * ( -1:(o.voxSize(2)+1) );
            grid on;
        end
    end
end

