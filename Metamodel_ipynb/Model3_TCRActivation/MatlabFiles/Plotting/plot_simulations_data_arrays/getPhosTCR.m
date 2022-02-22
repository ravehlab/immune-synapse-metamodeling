function [ptcr_x,ptcr_y] = ...
    getPhosTCR(linind_tcr,lck_phos_array,phos_threshold)

%{
Making a binary random selection of tcrs based on
the the lck probability distribution array.
 
Input: plots_parameters, Z1, tcr, lck_phos_array.
Output: tcr_binary_phos_x, tcr_binary_phos_y.
%}
    
% tcr locations array:
tcr_array = zeros(size(lck_phos_array));
tcr_array(linind_tcr) = 1;



% lck_phos_array multiplied by rand_array:
rand_array = rand(size(lck_phos_array));
rand_over_lck_phos_array = rand_array.*lck_phos_array;

% rand binary array based on rand_over_lck_phos_array: 
lck_phos_rand_binary_array = zeros(size(lck_phos_array));
lck_phos_rand_binary_array(rand_over_lck_phos_array > ...
    phos_threshold) = 1;

% ptcr_binary_array is the intersection of 
% 'lck_phos_rand_binary_array' locations and the 
% '' loctcr_arrayations:
ptcr_binary_array = lck_phos_rand_binary_array.*tcr_array;

% x,y coordinates of ptcr:
[ptcr_x,ptcr_y] = ...
    find(ptcr_binary_array);

end