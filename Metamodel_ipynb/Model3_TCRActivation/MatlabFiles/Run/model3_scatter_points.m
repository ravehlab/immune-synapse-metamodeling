% model3_scatter_points
clc
clear
%% doc: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
- generates randomly scattered points in an array.
- 
input: []
output:
calling: orangeBlueColormap(N),
         
%}
%
%% defining colors: %%%%%%%%%%%%%%%%%%%
TCR_color  = [0.0, 0.6, 0.0];
CD45_color = [1.0, 0.0, 0.0];
aLCK_color = [1.0, 0.0, 1.0];
pTCR_color = [1.0, 0.7, 0.0];

orange_blue_colormap = orangeBlueColormap(64);
%
%% array sizes: %%%%%%%%%%%%%%%%%%%%%%%
pixel_size = 10; % nm
array_size_x_pixels = 100; % 200
array_size_y_pixels = 100; % 200

array_size_x_microns = array_size_x_pixels*pixel_size/1000;
array_size_y_microns = array_size_y_pixels*pixel_size/1000;

limits1 = [0 array_size_x_pixels...
           0 array_size_y_pixels]; % used for plotting
       
%
%% generate TCR and CD45 locations: %%%
TCR_cluster_density = 1000;
CD45_cluster_density = 1000;

TCR_cluster_diameter_nm = 500;
CD45_cluster_diameter_nm = 500;

TCR_gaussian_cluster_sigma_nm = 80;
CD45_gaussian_cluster_sigma_nm = 80;

TCR_total_density = 300;
CD45_total_density = 300;

%
distribution = 'gaussian'; %'gaussian'; % 'square'
ind1 = 3; %1 2 3
%% square distribution: %%%%%%%%%%%%%%%
if strcmp(distribution,'square')
    %% TCR square: %%%%%%%%%%%%%%%%%%%%%%%%

    box_x0 = [0 1 1 0 0];
    box_y0 = [0 0 1 1 0];

    TCR_px = (1/ind1)*box_x0*array_size_x_pixels;
    TCR_py = box_y0*array_size_y_pixels;

    [TCR_x_pixels,TCR_y_pixels] = ...
        squareCluster(array_size_x_pixels,array_size_x_pixels,...
        TCR_total_density,pixel_size,TCR_px,TCR_py);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% CD45 square: %%%%%%%%%%%%%%%%%%%
    CD45_px = array_size_x_pixels -...
        (1/ind1)*box_x0*array_size_x_pixels;
    CD45_py = box_y0*array_size_y_pixels;

    [CD45_x_pixels,CD45_y_pixels] = ...
        squareCluster(array_size_x_pixels,array_size_x_pixels,...
        CD45_total_density,pixel_size,CD45_px,CD45_py);
end
%% gaussian distribution: %%%%%%%%%%%%%
if strcmp(distribution,'gaussian')
    
    %% TCR gaussian: %%%%%%%%%%%%%%%%%%%%%%
    TCR_N = 250;
    TCR_mu_x = 30/ind1; % 50
    TCR_sigma_x = 30/ind1; % 50;

    TCR_mu_y = [];
    TCR_sigma_y = [];

    [TCR_x_pixels,TCR_y_pixels] = ...
        gaussianCluster(array_size_x_pixels,array_size_x_pixels,...
        TCR_N,pixel_size,...
        TCR_mu_x,TCR_sigma_x,...
        TCR_mu_y,TCR_sigma_y);
    %
    %% CD45 gaussian: %%%%%%%%%%%%%%%%%%%%%
    CD45_N = 250;
    CD45_mu_x = array_size_x_pixels - 30/ind1;
    CD45_sigma_x = 30/ind1;

    CD45_mu_y = [];
    CD45_sigma_y = [];

    [CD45_x_pixels,CD45_y_pixels] = ...
        gaussianCluster(array_size_x_pixels,array_size_x_pixels,...
        CD45_N,pixel_size,...
        CD45_mu_x,CD45_sigma_x,...
        CD45_mu_y,CD45_sigma_y);
    %
end
%% saving what: %%%%%%%%%%%%%%%%%%%%%%%
save_plots = true;
%% plotting points: %%%%%%%%%%%%%%%%%%%
figure(1)
plot(TCR_x_pixels,TCR_y_pixels,'.','Color',TCR_color)
hold on
plot(CD45_x_pixels,CD45_y_pixels,'.','Color',CD45_color)
hold off
axis(limits1)
axis equal
axis tight
axis off

%
%% save plotting points: %%%%%%%%%%%%%%
if save_plots
   save_name = ['points_',distribution,'_',num2str(ind1)];
   saveas(gcf,[save_name,'.png'])
end
%
%% % TCR_locations_array: %%%%%%%%%%%%%
TCR_locations_array = zeros(array_size_x_pixels,...
    array_size_x_pixels);

linind_TCR = sub2ind(size(TCR_locations_array),...
    TCR_x_pixels,TCR_y_pixels);

TCR_locations_array(linind_TCR) = 1;
%
%% % CD45_locations_array: %%%%%%%%%%%%
CD45_locations_array = zeros(array_size_x_pixels,...
    array_size_x_pixels);

linind_CD45 = sub2ind(size(CD45_locations_array),...
    CD45_x_pixels,CD45_y_pixels);

CD45_locations_array(linind_CD45) = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% decay discs: %%%%%%%%%%%%%%%%%%%%%%%
R_max = ceil(array_size_x_pixels/2); % pixels
%% CD45: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% radius of dephosphorylation of TCR by CD45:
CD45_decay_length_nm = 10; % nm

% 2D distribution array describing the dephosphorylation
% probability of TCR by CD45:
CD45_decay_disk = decayDisk(...
    CD45_decay_length_nm,pixel_size,R_max);

norm_CD45_decay_disk = CD45_decay_disk/sum(sum(CD45_decay_disk));
%
%% aLCK: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
decay_length_nm = 20; %100; %20; %100 500
% radius of phosphorylation of TCR by aLck:
aLck_decay_length_nm = decay_length_nm; %80; %100;

% 2D distribution array describing the phosphorylation
% probability of TCR by aLck:
aLck_decay_disk = decayDisk(...
    aLck_decay_length_nm,pixel_size,R_max);

norm_aLck_decay_disk = aLck_decay_disk/sum(sum(aLck_decay_disk));
%
%% sum of decays: %%%%%%%%%%%%%%%%%%%%%
% phosphotylation distribution minus
% dephosphotylation distribution. their sum connot be
% negative.
sum_norm_decay_disk = norm_aLck_decay_disk - norm_CD45_decay_disk;
sum_norm_decay_disk(sum_norm_decay_disk < 0) = 0;
%
%% plot decay_probability_array: %%%%%%
figure(2)
h = surf(sum_norm_decay_disk');
h.EdgeColor = 'none';
axis(limits1)
axis equal
axis tight
axis off
view(2)
%% save decay disk: %%%%%%%%%%%%%%%%%%%
if false %save_plots
   save_name = ['decay_length_',num2str(decay_length_nm)];
   saveas(gcf,[save_name,'.png'])
end
%
%% aLck_probability_array: %%%%%%%%%%%%
% adding the distributions around all the CD45
% locaions:
decay_probability_array = aLckProbabilityArray(...
    sum_norm_decay_disk,...
    array_size_x_pixels,array_size_y_pixels,...
    CD45_x_pixels,CD45_y_pixels);
%
%% plot decay_probability_array: %%%%%%
figure(3)
h = surf(decay_probability_array'-1);
h.EdgeColor = 'none';
hold on
plot(TCR_x_pixels,TCR_y_pixels,'.','Color',TCR_color)
plot(CD45_x_pixels,CD45_y_pixels,'.','Color',CD45_color)
hold off
axis(limits1)
axis equal
axis tight
axis off
view(2)
%
%% save plot decay_probability_array: %
if save_plots
   save_name = ['decay_probability_array_',...
       distribution,num2str(ind1),'_decLen',...
       num2str(decay_length_nm),'nm'];
   saveas(gcf,[save_name,'.png'])
end
%
%% pTCR_probability_array:
pTCR_probability_array = decay_probability_array'.*...
    TCR_locations_array';
%
%% plot pTCR_probability_array: %%%%%%%
figure(4)
h = surf(pTCR_probability_array-1); % sum_norm_decay_disk
h.EdgeColor = 'none';
colormap(orange_blue_colormap);
axis(limits1)
axis equal
axis tight
axis off
view(2)
%
%% save plot decay_probability_array: %
if save_plots
   save_name = ['TCR_phos_probability_array_',...
       distribution,num2str(ind1),'_decLen',...
       num2str(decay_length_nm),'nm'];
   saveas(gcf,[save_name,'.png'])
end
%



