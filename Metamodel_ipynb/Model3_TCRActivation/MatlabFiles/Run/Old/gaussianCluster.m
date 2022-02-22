function [x_in,y_in] = gaussianCluster(sx,sy,N,pixel_size,...
    mu_x,sigma_x,mu_y,sigma_y)

%% doc: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
create random distribution of points with density 'density'
inside a polygon (px,py). The polygon is inside a
rectangular array of pixels [0 sx 0 sy].

input: sx = array size x (pixels).
       sy = array size y (pixels).
       pixel_size = pixel size (nm).
       px = x cordinates of polygon (pixels).
       py = x cordinates of polygon (pixels).
       density = number of points per square micron.
calls: [].
output: x = x coordinate of points locations.
        y = y coordinate of points locations.
%}
%
% N = 250;
if ~isempty(mu_x) && ~isempty(sigma_x)
    x = mu_x + sigma_x*randn(N,1);
else
    x = sx*rand(N,1);
end
x = ceil(x);

if ~isempty(mu_y) && ~isempty(sigma_y)
    y = mu_y + sigma_y*randn(N,1);
else
    y = sy*rand(N,1);
end
y = ceil(y);

%% selecting points in the array limits:
px = [1 sx sx 1 1];
py = [1 1 sy sy 1];
in = inpolygon(x,y,px,py);

x_in = x(in);
y_in = y(in);


end