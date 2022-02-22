function gr = ...
    GR(array1,array2,R_max)


gr = zeros(1,R_max+1);

for ring_radius = 0:R_max
    if ring_radius == 0
        ring_mask = 1;
    else
        [X,Y] = meshgrid(-ring_radius:ring_radius);
        ring_mask = floor(sqrt(X.^2 + Y.^2)) == ring_radius;
    end
    ring_area = sum(sum(ring_mask));

    % putting the ring on the full array:
    rings_conv_array1 = conv2(array1, ring_mask, 'same');

    % linear index of the ring point on the full array:
    linind_array2 = find(array2);

    % sum values on ring:
    values_on_ring = rings_conv_array1(linind_array2);

    % sum devided by ring area:
    gr(ring_radius+1) = sum(values_on_ring)/ring_area;

end

fill_factor2 = sum(array2(:))/numel(array2);
gr = gr/fill_factor2;

end


