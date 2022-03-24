function PAR = InitSPH(KRAD, NUM, DIM)
%INITSPH Initializes particles of dam break model
%   initialize at the limits of DIM=[xmin,xmax;ymin,ymax]
RAND_MAX = 50;

pos_x = (repelem((1:NUM)*KRAD, NUM)') + KRAD*randn(NUM^2,1)/RAND_MAX;
pos_y = (repmat((1:NUM)*KRAD, [1,NUM])') + KRAD*randn(NUM^2,1)/RAND_MAX;

pos = [pos_x, pos_y] - [mean(pos_x), mean(pos_y)] + ...
    [.5*(DIM(1,2)-DIM(1,1)), .4*(DIM(2,2)-DIM(2,1))];

PAR = Particle(pos);

PAR.lim = DIM;
PAR.rad = KRAD;

end

