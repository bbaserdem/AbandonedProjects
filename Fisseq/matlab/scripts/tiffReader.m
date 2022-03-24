function [ I ] = tiffReader( directory, varargin )
% IF_TIFFREADER Reads tiff from iarpa fisseq data
% Can only read one base at a time

if isempty(directory)   % Allow reading direcly from directory if no input
    directory = [ pwd, '/' ];
end

if strcmp( directory(end-3:end), '.tif' )
    % Use if the 1 base image is in a z stack with consecutive channels
    flag = 1;
    I = imfinfo( directory );
    N = length(I);
    lims = { [1, 1, I(1).Height], [1, 1, I(1).Width] };
else
    % Use if the 1 base image is contained in a folder of 2d images
    flag = 2;
    if directory(end) ~= '/'
        directory = [ directory, '/'];
    end
    files = dir( [directory, '*.tif'] );
    N = length(files);
    I = imfinfo( [directory, files(1).name] );
    lims = { [1, 1, I.Height], [1, 1, I.Width] };
end

ch = false;
prom = true;

for var = 1:(nargin-2)
    elseif strcmp( varargin{var}, 'Range')
        % Expect limits as a cell array to be passed to imread;
        % { [ Ymin, (Yskip), Ymax ] ,   ROWS
        %   [ Xmin, (Xskip), Xmax ] }   COLUMNS
        if       ~iscell( varargin{var+1} )                         || ...
                  length(varargin{var+1})~=2                        || ...
                ( length(varargin{var+1}{1})~=2                 &&     ...
                  length(varargin{var+1}{1})~=3 )                   || ...
                ( length(varargin{var+1}{2})~=2                 &&     ...
                  length(varargin{var+1}{2})~=3 )                   || ...
                ( varargin{var+1}{1}(1) < 1)                        || ...
                ( varargin{var+1}{1}(1) > lims{1}(end) )            || ...
                ( varargin{var+1}{1}(end) < varargin{var+1}{1}(1) ) || ...
                ( varargin{var+1}{1}(end) > lims{1}(end) )          || ...
                ( varargin{var+1}{2}(1) < 1)                        || ...
                ( varargin{var+1}{2}(1) > lims{2}(end) )            || ...
                ( varargin{var+1}{2}(end) < varargin{var+1}{2}(1) ) || ...
                ( varargin{var+1}{2}(end) > lims{2}(end) )
            error('Invalid range');
        end
        lims = varargin{var+1};
    elseif strcmp( varargin{var}, 'Channels' )
        % Expect channel argument as an integer that divides length (N)
        ch = varargin{var+1};
        if mod(N, ch) ~= 0
            error( 'Invalid channel no' );
        end
    elseif strcmp( varargin{var}, 'Prompt' )
        if islogical varargin{var+1}
            prom = varargin{var+1};
        else
            error('"Prompt" field requires boolean value.')
        end
    end
end

if length(lims{1}) == 2
    R = length(lims{1}(1) : lims{1}(2) );
elseif length(lims{1}) == 3
    R = length(lims{1}(1) : lims{1}(2) :lims{1}(3) );
end
if length(lims{2}) == 2
    C = length(lims{2}(1) : lims{2}(2) );
elseif length(lims{1}) == 3
    C = length(lims{2}(1) : lims{2}(2) :lims{2}(3) );
end

I = zeros(R, C, N, 'uint16');
fprintf('\n');
PROMPT = 12;
P_COUNT=PROMPT;
for a=1:N
    if flag == 1
        I(:,:,a) = imread( directory, 'Index', a,     'PixelRegion', lims);
    elseif flag == 2
        I(:,:,a) = imread([directory, files(a).name], 'PixelRegion', lims);
    end
    
    if prom
        if ( mod(a,PROMPT) == 1 )
            if a ~= 1
                fprintf('    (/%3d)', N);
            end
            
            fprintf('\n');
            P_COUNT=PROMPT;
        end
        fprintf('%3d ', a);
    end
    
    P_COUNT = P_COUNT - 1;
end

if prom
    for b = 1 : ( P_COUNT + 1 )
        fprintf('    ');
    end
    fprintf('( OK )\n');
end

if ch
    I = reshape(I, R, C, N/ch, ch);
end

end

