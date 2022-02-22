function exponential_decay_kernel = ...
    ExponentialDecayKernel(decay_length_nm,pixel_size_nm,Display)

%{
Active Lck probability distribution around a single CD45.
Input: decay_length_nm, pixel_size_nm, Display.
Output: exponential_decay_kernel.
%}

decay_length_pixels = round(decay_length_nm/pixel_size_nm);

% Decay kernel in one dimension. 2*kernel_size/decay_length.
K2DL_ratio = 5;
kernel_x_pixels = -K2DL_ratio*decay_length_pixels:1:...
    K2DL_ratio*decay_length_pixels;

[kernel_X_pixels,kernel_Y_pixels] = meshgrid(kernel_x_pixels);
kernel_R_pixels = sqrt(kernel_X_pixels.^2 + kernel_Y_pixels.^2);
kernel_Z = exp(-kernel_R_pixels/decay_length_pixels);

% normalized probability distribution:
exponential_decay_kernel = kernel_Z/sum(sum(kernel_Z));

if Display
    figure(71)
    h = surf(exponential_decay_kernel);
    h.EdgeColor = 'none';
%     colormap(plots_parameters.colormaps.magenta_fixed)
    grid off
    box on
    alpha color
    alpha scaled
end
    
end