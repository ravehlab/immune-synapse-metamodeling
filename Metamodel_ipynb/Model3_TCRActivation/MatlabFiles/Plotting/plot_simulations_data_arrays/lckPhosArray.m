function lck_phos_array = lckPhosArray(linind_cd45,Z1,...
    decay_length_nm,off_disk_radius_nm,pixel_size_nm)

cd45_array = zeros(size(Z1)); % array size
cd45_array(linind_cd45) = 1; % cd45 locations

decay_length_pixels = decay_length_nm/pixel_size_nm; 

%% Lck distribution
% Define Lck probability distriution around a single cd45. 
decay_mask_x = -5*decay_length_pixels:1:5*decay_length_pixels;
[decay_mask_X,decay_mask_Y] = meshgrid(decay_mask_x);
decay_mask_R_pixels = sqrt(decay_mask_X.^2 + decay_mask_Y.^2);
decay_mask_Z = exp(-decay_mask_R_pixels/decay_length_pixels);
decay_mask_Z = decay_mask_Z/sum(sum(decay_mask_Z));
% Lck probability all over the array.
Lck_array = conv2(cd45_array,decay_mask_Z,'same');

% tcr phosphorylation 'on' distribution.
phos_on_array = Lck_array;

% tcr phosphorylation 'off' distribution.
off_disk_radius_pixels = off_disk_radius_nm/pixel_size_nm;
h_disk = fspecial('disk',off_disk_radius_pixels);               
phos_off_array = imfilter(cd45_array,h_disk,'replicate');

% Combined 'on' and 'off' distributions.
phos_ability_array = phos_on_array;
% phos_ability_array(logical(phos_off_array)) = 0;

% normalizing the 'phos_ability_array' to max = 1.
norm_phos_ability_array = phos_ability_array/...
    max(max(phos_ability_array));

lck_phos_array = norm_phos_ability_array;

if 1
    hsize = 5;
    sigma_smooth = 1;
    h = fspecial('gaussian',hsize,sigma_smooth);
    smoothed_lck_phos_array = ...
        imfilter(lck_phos_array,h,'replicate');

    lck_phos_array = smoothed_lck_phos_array;
end

end