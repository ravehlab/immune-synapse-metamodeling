function alck_array = aLckArray(...
    cd45_array,decay_length_pathces)

% Define Lck probability around a single cd45.
% Decay in x direction:
decay_mask_x = -5*decay_length_pathces:1:5*decay_length_pathces;
[decay_mask_X,decay_mask_Y] = meshgrid(decay_mask_x);

% Radial decay mask:
decay_mask_R_pixels = sqrt(decay_mask_X.^2 + decay_mask_Y.^2);
decay_mask_Z = exp(-decay_mask_R_pixels/decay_length_pathces);

% Lck distribution over the all array:
alck_array = conv2(cd45_array,decay_mask_Z,'same');

end