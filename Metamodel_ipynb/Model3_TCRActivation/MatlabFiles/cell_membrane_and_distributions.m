% cell_membrane_and_distributions
% aim: 
% 1. Create a smoothed random t-cell membrane.
% 2. Scatter points randomly according hights
% and other relevant data.

array_size_nm = 2000;
pixel_size_nm = 10;

array_size_pixels = round(array_size_nm/pixel_size_nm);

array0 = ones(array_size_pixels,array_size_pixels);
z_max_nm = 70;
array_z_max_nm = z_max_nm*array0;

array_z0 = array_z_max_nm;
%% Set membrane height:
array_z0(20,20) = 13;
array_z0(50,30) = 13;
array_z0(100,170) = 50;

%% smooth membrane:
hsize = 101;
sigma = 3;
h = fspecial('gaussian',hsize,sigma);
array_f = imfilter(array_z0,h,'replicate'); 
% dz1 = abs(z_max_nm - 13);
array_z = 70+57*(array_f-70)/(70-min(min(array_f)));

%
figure(4)
h1 = pcolor(array_z);
set(h1, 'EdgeColor','none')
axis equal
axis tight
colorbar
colormap(gray)

hold on
levels = 0:10:60;
[M,c] = contour(array_z, levels);
c.Color = [0.0 0.0 0.0];

hold off
caxis([0 70])