clear variables

T = 200;
LW = 2;
PW = 100;

% Time parameter
t = linspace( 0, 1, T )';

% First function (Make sure ends at the same spot)
x1 = .3 * t.^2 + .2 * sin( 2*pi*t );
y1 = cos( 3*pi*t );
z1 = t;
c1 = [30,144,255] / 255;

% Second function (Make sure starts and ends at the same spots)
x2 = .3 * sqrt(t) + .25 * sin( 5*pi*t );
y2 = cos( 5*pi*t );
z2 = t.^2;
c2 = [255,69,0] / 255;


figure(1);

% X plot
subplot(3,2,1)
plot(t,x1,'Color',c1,'LineWidth',LW);
hold on
plot(t,x2,'Color',c2,'LineWidth',LW);
sx = scatter( [t(1);t(1)], [x1(1);x2(1)], PW*[1;1], [c1;c2],'filled');
hold off
title( 'X through time' );
xlabel( 'Time' );
ylabel( 'X value' );

% Y plot
subplot(3,2,3)
plot(t,y1,'Color',c1,'LineWidth',LW);
hold on
plot(t,y2,'Color',c2,'LineWidth',LW);
sy = scatter( [t(1);t(1)], [y1(1);y2(1)], PW*[1;1], [c1;c2],'filled');
hold off
title( 'Y through time' );
xlabel( 'Time' );
ylabel( 'Y value' );

% Z plot
subplot(3,2,5)
plot(t,z1,'Color',c1,'LineWidth',LW);
hold on;
plot(t,z2,'Color',c2,'LineWidth',LW);
sz = scatter( [t(1);t(1)], [z1(1);z2(1)], PW*[1;1], [c1;c2],'filled');
hold off
title( 'Z through time' );
xlabel( 'Time' );
ylabel( 'Z value' );

% Dimensional plot
subplot(3,2,[2,4,6])
plot3(x1,y1,z1,'Color',c1,'LineWidth',LW);
hold on
plot3(x2,y2,z2,'Color',c2,'LineWidth',LW);
sd = scatter3( [x1(1);x2(1)], [y1(1);y2(1)], [z1(1);z2(1)], PW*[1;1], [c1;c2],'filled');
hold off
title( 'XYZ within space' );

for i = 1:T
    % Modify X plot
    sx.XData(:) = t(i);
    sx.YData = [ x1(i), x2(i) ];
    % Modify Y plot
    sy.XData(:) = t(i);
    sy.YData = [ y1(i), y2(i) ];
    % Modify Z plot
    sz.XData(:) = t(i);
    sz.YData = [ z1(i), z2(i) ];
    % Modify dimension plot
    sd.XData = [ x1(i), x2(i) ];
    sd.YData = [ y1(i), y2(i) ];
    sd.ZData = [ z1(i), z2(i) ];
    
    drawnow;
end