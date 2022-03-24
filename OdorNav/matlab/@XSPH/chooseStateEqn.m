function chooseStateEqn(o, EQN, varargin)
%CHOOSESTATEEQN Hard code the equation of state

switch EQN
    case {'tait', 'Tait'}
        gamma = varargin{1};
        o.pressure = @(x) math.tait(        x, o.d0, o.c0, gamma);
        o.density =  @(x) math.tait_inverse(x, o.p0, o.c0, gamma);
    case {'ideal gas'}
        o.pressure = @(x) math.tait(        x, o.d0, o.c0, 1);
        o.density =  @(x) math.tait_inverse(x, o.p0, o.c0, 1);
    otherwise
        error('Equation of state not recognized');
end

end

