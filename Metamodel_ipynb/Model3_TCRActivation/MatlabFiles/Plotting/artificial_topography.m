% artificial_topography
z_max = 70; % nm
array_size = 201;
z_array0 = z_max*ones(array_size,array_size);

xx = -100:100;
[X,Y] = meshgrid(xx);

z_array = z_array0;
z_array(X.^2 + Y.^2 < 50^2) = 13;
z_array(X.^2 + Y.^2 > 75^2 &...
        X.^2 + Y.^2 < 95^2) = 50;


figure(11)
imagesc(z_array)




