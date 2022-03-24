function P = particleInBox
%PARTICLEINBOX Test case with a single particle in a box

P = init.lidDrivenCavity(50,1,0);
P.Bv(:) = 0;
P.x = rand(1,2);
P.v = rand(1,2);
P.u = P.v;
P.f = zeros(1,2);
P.s = P.s(1);
P.d = P.d(1);
P.m = P.m(1);
P.p = P.p(1);
P.n = 0;
P.h = .1;
P.sup = .2;
P.pressure = @(x) x*0;

end

