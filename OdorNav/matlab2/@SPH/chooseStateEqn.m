function chooseStateEqn(o, EQN, varargin)
%CHOOSESTATEEQN Hard code the equation of state

switch EQN
    case {'tait', 'Tait'}
        gamma = varargin{1};
        o.sim_state = ['tait \gamma=', num2str(gamma)];
        o.int_pre = @(x) math.tait(        x, ...
            o.par_den, o.par_snd, gamma);
        o.int_den = @(x) math.tait_inverse(x, ...
            o.par_pre, o.par_snd, gamma);
    case {'ideal gas'}
        o.sim_state = 'ideal gas';
        o.int_pre = @(x) math.tait(        x, ...
            o.par_den, o.par_snd, 1);
        o.int_den = @(x) math.tait_inverse(x, ...
            o.par_den, o.par_snd, 1);
    otherwise
        error('Equation of state not recognized');
end

end

