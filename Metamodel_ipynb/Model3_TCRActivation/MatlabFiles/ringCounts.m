function [rings_counts,normalyzed_rings_counts] = ...
    ringCounts(array_size,linind_points,s_points)

array0 = zeros(array_size);
xx = -floor(array_size(1)/2)+1:1:floor(array_size(1)/2);
[XX,YY] = meshgrid(xx);

R_max = floor(array_size(1)/2);
rings_areas = zeros(1,R_max);
rings_counts = zeros(1,R_max);

for r = 1:R_max
    
    disk_r1 = array0;
    disk_r2 = array0;

    disk_r1(XX.^2 + YY.^2 < r^2) = 1;
    disk_r2(XX.^2 + YY.^2 < (r+1)^2) = 1;

    ring_r = disk_r2 - disk_r1;
    ring_area = sum(sum(ring_r));
    rings_areas(r) = ring_area;
    
    ring_counts = sum(ring_r(linind_points).*s_points);
    rings_counts(r) = ring_counts;
    
end

normalyzed_rings_counts = rings_counts./rings_areas;

end