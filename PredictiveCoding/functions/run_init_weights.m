function [ n ] = run_init_weights( n )
%GENERATE_NETWORK Initializes weights of a full network

n.L_no = length( n.Xsize );
n.X_no = length( n.Xsize );
n.D_no = length( n.Dsize );
n.W = cell(n.L_no,1);
n.B = cell(n.L_no,1);
n.A = cell(n.L_no,1);

% Uses Xavier initialization. May tweak the factor
for a = 1:n.L_no
    n.W{a} = randn( n.Xsize(a),   n.Dsize(a) ) * sqrt( ...
        n.init_k / ( n.alpha * (n.Xsize(a) + n.Dsize(a)) ) );
    n.A{a} = randn( n.Dsize(a+1), n.Xsize(a) ) * sqrt( ...
        n.init_k / ( n.alpha * (n.Xsize(a) + n.Dsize(a+1)) ) );
    n.B{a} = zeros( n.Xsize(a), 1);
end

if isfield( n, 'fix_end' )
    if n.fix_end
        if all( n.Xsize == n.Dsize(2:end) )
            for b = n.L_no
                n.A{b} = eye( n.Xsize(b) );% / sqrt( n.alpha );
            end
        else
            fprintf( [ '\nTo fix end connection to identity, ', ...
                'last X and D layers must be of same size!\n', ...
                'Defaulting to regular learning . . .'] );
            n.fix_end = false;
        end
    end
else
    n.fix_end = false;
end

end

