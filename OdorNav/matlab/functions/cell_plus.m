function o = cell_plus(o,varargin)
%CELL_PLUS Adds all inputs
for d = 1:length(varargin)
    o = cellfun(@plus, o, varargin{d}, 'UniformOutput', false);
end
end

