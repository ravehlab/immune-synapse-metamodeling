function tcr_array = tcrArray(A0,density,R_patches)

N_patches = numel(A0);
A0_sq_microns = N_patches/10000;
N_tcr_total = A0_sq_microns*density;

% randomley selecting 'N_tcr_total' linear indices from 
% 'N_patches' possibilities with no repetitions.
linind_tcr_total = ...
    randsample(N_patches,N_tcr_total,false);

% inintialize tcr_array
tcr_array = A0;
% random lininds all over the array = 1
tcr_array(linind_tcr_total) = 1;

% Deleting points outside circle with radius R 
% around the center:
center_x_patches = round(size(A0,2)/2);
center_y_patches = round(size(A0,1)/2);

x_patches = 1:size(A0,2);
y_patches = 1:size(A0,1);
[X_patches,Y_patches] = meshgrid(x_patches,y_patches);

tcr_array((X_patches - center_x_patches).^2 + ...
          (Y_patches - center_y_patches).^2 > ...
           R_patches^2) = 0;

end