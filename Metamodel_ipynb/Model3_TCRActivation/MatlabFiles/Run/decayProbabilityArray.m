function decay_probability_array = decayProbabilityArray(...
    decay_disk,array_size_x_pixels,array_size_y_pixels,...
    CD45_x_pixels,CD45_y_pixels)

%% doc: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
Create a decay probablity array by doing a 2D 
convolution (conv2) between an array of CD45 points
and a 2D array of exponential decay distribution.

Input: decay_disk = 2D array of decay distribution.
       array_size_x_pixels = points array x size (pixels).
       array_size_y_pixels = points array y size (pixels).
       CD45_x_pixels = CD45 x locations (pixels).
       CD45_y_pixels = CD45 y locations (pixels).
Calls: []
Output: decay_probability_array = 2D array with sizes
        (array_size_x_pixels x array_size_y_pixels).
%}

%% make array form CD45 locations: %%%%
% initialize array:
CD45_locations_array = ...
    zeros(array_size_x_pixels,array_size_y_pixels);

% x,y locations to linear index:
linind_CD45 = sub2ind(size(CD45_locations_array),...
    CD45_x_pixels,CD45_y_pixels);

% assign 1's to locations of molecules:
CD45_locations_array(linind_CD45) = 1;

% 2D convolution:
probability_sum = ...
    conv2(CD45_locations_array, decay_disk, 'same');

decay_probability_array = probability_sum;

end