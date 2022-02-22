function [r,Rg] = radiusOfGyration(x,y)

mean_x = mean(x);
mean_y = mean(y);
N = length(x);

r_squared = ((x - mean_x).^2 + (y - mean_y).^2);
r = sqrt(r_squared);

Rg = sqrt(sum(r_squared./N));

end