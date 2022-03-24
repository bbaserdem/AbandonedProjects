% Check if things are working

p1 = rand( 50, 2 );
r1 = randperm( size(p1,1) );
p2 = rand( 100, 2 );
r2 = randperm( size(p2,1) );


dst = zeros( size(p1,1), size(p2,1) );
for d = 1:size(p1,2)
    dst(r1,r2) = dst(r1,r2) + ( p1(:,d) - p2(:,d)' ).^2;
end
dst = sqrt( dst );

sq1 = dot( p1 , p1 , 2 );
sq2 = dot( p2 , p2 , 2 );
dstCalc(r1,r2) = sqrt( sq1 + sq2' - 2 * ( p1 * (p2') ) );

imagesc( dstCalc - dst );
colorbar;