function [ RES ] = fisseq_detect( I, detVar, imVar)
% FISSEQ_DETECT Main wrapper function to run detection algorithm
%   This function mainly does following;
%     - Take sections off image
%     - Sends image to analysis pipeline
%       -- Remove background
%       -- Send

% Takes in following arguments (all values have defaults, other than "I")
%   I:              Image to work on, can be the path to tiff image(s)
%   detVar:         Scalar struct, containing following parameters;
detVariables = {
    'splitOverlap'          '50'    % Overlap size for image splitting
    'minimalRegSize'        '10'    % Minimal WS region size to consider
    'bootstrapRadius'       '5'     % Region radius to limit BS statistics
    'lowPassFilterRadius'   '0'     % Filter radius to eliminate noise (WS)
    'highPassFilterRadius'  '4'     % Filter radius to eliminate background
    'c1DirMinSigma'         '1'     % Values to restrict the detected pts
    'c1DirMaxSigma'         '6'     %   These are XY plane sigma limits
    'c2DirMinSigma'         '1'     % 
    'c2DirMaxSigma'         '6'     % 
    'zDirMinSigma'          '0'     %   These are Z axis sigma limits
    'zDirMaxSigma'          'Inf'   % 
    'intensityMin'          '1000'  % These are intensity limits for disc
    'intensityMax'          'Inf'   %   potential rolonies
    'plotFlag'              'true'  % Flag to plot results
};% imVar:          Scalar struct, containing the following parameters
imVariables = {
    'imName'            '''I'''     % Name of image varuable in mat file
    'splitSize'         '10000'     % Grid size to split image
    'bootstrapTrials'   '30'        % Number of trials for bootstrap
    'dyeBleed'          'eye(4)'    % Dye bleed matrix
    'map'               '''GTAC'''  % Channel to base map
    'genes'             '[]'        % Crossref library (struct: id,seq)
};

% Returns the following
RES = struct( 'X', {}, 'Y', {}, 'Z', {}, 'id', {} );
SEQ = struct( 'C', {}, 'sequence', {}, 'frequency', {} );

%-----LOAD VARIABLES-----%

for a = 1:size( detVariables, 1 )
    if exist('detVar', 'var')
        if isfield('detVar', detVariables{a,1})
            eval( [ detVariables{a,1}, '=', ...
                'detVar.', detVariables{a,1}, ';' ] );
            continue
        end
    end
    eval( [ detVariables{a,1}, '=', detVariables{a,2}, ';' ] );
end
for a = 1:size( imVariables, 1 )
    if exist('imVar', 'var')
        if isfield('imVar', imVariables{a,1})
            eval( [ imVariables{a,1}, '=', ...
                'imVar.', imVariables{a,1}, ';' ] );
            continue
        end
    end
    eval( [ imVariables{a,1}, '=', imVariables{a,2}, ';' ] );
end

% Extract image dimensions
image = matfile(I);

[xSize,ySize,zSize,channelNo,baseNo] = size( image, imName );

% Task splitting
xSVN = ceil(xSize / splitSize);
ySVN = ceil(ySize / splitSize);
for xSV = 1:xSVN
    for ySV = 1:ySVN        % Super voxel indices
        fprintf('region (%d, %d) of (%d, %d)\n', xSV, ySV, xSVN, ySVN);
        xRange = [ max( 1, 1+(xSV-1)*splitSize-splitOverlap), ...
            min(xSize, xSV*splitSize+splitOverlap)]
        yRange = [ max( 1, 1+(ySV-1)*splitSize-splitOverlap), ...
            min(ySize, ySV*splitSize+splitOverlap)]
        
        % Get cutout
        IM = double( image.(imName)(...
            xRange(1):xRange(2),...
            yRange(1):yRange(2),...
            1:zSize,...
            1:channelNo,...
            1:baseNo ) );
        
        size(IM)
    end
end










































end