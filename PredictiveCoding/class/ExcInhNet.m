classdef ExcInhNet
    %EXCINHNET Class structure for simulating the antisymmetric PC net.
    
    
    %-----PROPERETY SETTING-----%
    properties
        LearnRate
        Decay
        Train
        Test
        TimeConst
    end
    properties(Dependent)
        LayerNo
        CompleteFlag
    end
    properties(SetAccess=protected)
        Layers
        X
        D
        W
        A
        B
        Nonlin
        BiasOn
    end
    properties(GetAccess=protected, SetAccess=protected)
        CompleteFlag
    end
    
    
    %-----METHODS SETTING-----%
    methods
        
        function obj = ExcInhNet(layer, func)
            % Class instantiator.
            % These values should be left default.
            obj.LearnRate = 0.001;
            obj.Decay = 0;
            obj.TimeConst = Inf;
            if nargin > 1
                % Nonlinearity. For now, just support linear
                switch func
                    case 'linear'
                        obj.Nonlin = @(x) x;
                        obj.NonlinDeriv = @(x) ones(size(x));
                    otherwise
                        if isa(func,'function_handle')
                            obj.Nonlin = func;
                        else
                            warning('Invalid function type, using linear');
                            obj.Nonlin = @(x) x;
                            obj.NonlinDeriv = @(x) ones(size(x));
                        end
                end
            end
            if nargin > 0
                % Initialize weights
                obj.Layers = layer;
                obj.W = cell(1, obj.LayerNo);
                for a = 1:(obj.LayerNo)
                    obj.W{a} = sqrt(2/(obj.Layers(a)+obj.Layers(a+1))) *...
                        randn( obj.Layers(a), obj.Layers(a+1) );
                end
            end
        end
        
        %-----DEPENDENT METHODS-----%
        function LayerNo=get.LayerNo(obj)
            % Get the size of layers
            LayerNo = length(obj.Layers) - 1;
        end
        function CompleteFlag=get.CompleteFlag(obj)
            if isempty( obj.LearnRate ) || ...
                    isempty( obj.TimeConst ) || ...
                    isempty( obj.Layers ) || ...
                    isempty( obj.W ) || ...
                    isempty( obj.Nonlin ) || ...
                    isempty( obj.B )
                CompleteFlag = false;
            else
                CompleteFlag = true;
            end
        end
    end
    
end

