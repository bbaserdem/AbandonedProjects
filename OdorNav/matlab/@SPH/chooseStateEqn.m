function chooseStateEqn(o, EQN, varargin)
%CHOOSESTATEEQN Hard code the equation of state

switch EQN
    case {'tait', 'Tait'}
        gamma = varargin{1};
        o.stateEquation = 'tait';
        o.fn_pre = @(x) math.tait(        x, ...
            o.refDensity, o.sound, gamma);
        o.fn_den = @(x) math.tait_inverse(x, ...
            o.refPressure, o.sound, gamma);
    case {'ideal gas'}
        o.stateEquation = 'ideal gas';
        o.fn_pre = @(x) math.tait(        x, ...
            o.refDensity, o.sound, 1);
        o.fn_den = @(x) math.tait_inverse(x, ...
            o.refDensity, o.sound, 1);
    otherwise
        error('Equation of state not recognized');
end

end

