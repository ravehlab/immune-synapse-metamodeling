function [x_in,y_in] = squareCluster(...
    sx,sy,density,pixel_size,px,py)

%% doc: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
create random distribution of points with density 'density'
inside a polygon (px,py). The polygon is inside a
rectangular array of pixels [0 sx 0 sy]. pixel size is
'pixel_size'.

input: sx = array size x (pixels).
       sy = array size y (pixels).
       pixel_size = pixel size (nm).
       px = x cordinates of polygon (pixels).
       py = x cordinates of polygon (pixels).
       density = number of points per square micron.
calls: 
output: x = x coordinate of points locations.
        y = y coordinate of points locations.
%}
%
%% pobability that a pixel will contain a point:
% a pixel area is (pixel_size)^2 micron^2. 
P_pixel = (pixel_size/1000)^2*density;
% P_array = zeros(sx,sy); % sx, sy = size in pixels
N_pixels = sx*sy;
N_all = ceil(N_pixels*P_pixel);

linind_points = randsample(N_pixels,N_all,false);
[x,y] = ind2sub([sx,sy],linind_points);

in = inpolygon(x,y,px,py);

x_in = x(in);
y_in = y(in);
%%

% linind_all = 1:N_all;
% s = RandStream('mlfg6331_64'); 
% linind_points = randsample(s,N_pixels,N_all,false);

%{
in = inpolygon(xq,yq,xv,yv);
[in,on] = inpolygon(xq,yq,xv,yv);


xq = randn(250,1);
yq = randn(250,1);

xq = rand(500,1)*5;
yq = rand(500,1)*5;

xv = [1 4 4 1 1 NaN 2 2 3 3 2];
yv = [1 1 4 4 1 NaN 2 3 3 2 2];
figure

plot(xv,yv) % polygon
axis equal

hold on
plot(xq(in),yq(in),'r+') % points inside
plot(xq(~in),yq(~in),'bo') % points outside
hold off
%}

end