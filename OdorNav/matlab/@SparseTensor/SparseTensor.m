classdef SparseTensor < handle
    %SPARSETENSOR Storage method for a tensorial quantity with sparse
    % elements
    
    properties
        data        % Data stored as cell of sparse arrays
        dim         % Dimension of tensorial index
        sdim        % Dimension of sparse index
    end
    
    methods
        function o = SparseTensor(varargin)
            %SPARSETENSOR Construct an instance of this class
            if nargin == 1
                % Construct directly from cell array
                if iscell(varargin{1})
                    % Check if all matrices are sparse
                    if ~all(cellfun(@issparse, varargin{1}(:)))
                        error('All entries of cell argument must be sparse');
                    end
                    % Check dimensions of all matrices
                    if range(cellfun(@(x) size(x,1), varargin{1}(:)))~=0
                        error('All entries of cell must be the same size')
                    end
                    if range(cellfun(@(x) size(x,2), varargin{1}(:)))~=0
                        error('All entries of cell must be the same size')
                    end
                    if issparse(varargin{1}{1})
                        o.data = varargin{1};
                        o.dim = size(o.data);
                        o.sdim = size(o.data{1});
                        return;
                    end
                end
                error('Single input must be a cell array of sparse matrices');
            elseif nargin >= 3
                if (~isvector(varargin{1})) || (~isvector(varargin{2}))
                    error('First two arguments should be sparsity index');
                end
                if all(size(varargin{1})==size(varargin{2}))
                    if nargin == 3
                        o.sdim = [max(varargin{1}(:)), max(varargin{2}(:))];
                    elseif nargin == 4
                        o.sdim = varargin{4};
                    elseif nargin == 5
                        o.sdim = [varargin{4},varargin{5}];
                    end
                    sto = size(varargin{3});
                    if length(sto)==2
                        o.dim = [sto(2), 1];
                    else
                        o.dim = sto(2:end);
                    end
                    o.data = cell(o.dim);
                    mat = reshape(varargin{3},size(varargin{3},1),[]);
                    for i = 1:size(mat,2)
                        o.data{i} = sparse(varargin{1}, varargin{2}, ...
                            mat(:,i), o.sdim(1), o.sdim(2) );
                    end
                    return;
                end
                error('Unsuitable input');
            end
        end
    end
end

