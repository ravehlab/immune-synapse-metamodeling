function [r,z] = xyZ2rz(x,y,Z)

%{
Input: x,y coordinates, Z array.
Output: r relative to array center, z(r).
%}
center_x = round(size(Z,2)/2);
center_y = round(size(Z,1)/2);

r = sqrt((x - center_x).^2 + (y - center_y).^2);
linind = sub2ind(size(Z),x,y);
z = Z(linind);


end