% plot_data_from_my_Razvag_figs
clear
clc
path1 = 'C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\MatlabFiles\Results\From_20210816\zap70_Razvag_new';
addpath(path1);
open('zap70_IRM_90_sec.fig');
a = get(gca,'Children');
xdata = get(a, 'XData');
ydata = get(a, 'YData');
zdata = get(a, 'ZData');

surf_zdata = zdata{2};
shifted_surf_zdata = surf_zdata - 115; % nm
contour_zdata = zdata{3};
%
%% cropping the point to point inside the topography area.
array_size = size(surf_zdata);
scatter_xdata = xdata{1};
scatter_ydata = ydata{1};

scatter_xy_data = [scatter_xdata',scatter_ydata'];
% crop_scatter_xy_data:
cropped_scatter_xy_data = scatter_xy_data(...
    scatter_xdata' > 0 & scatter_xdata' < array_size(1) &...
    scatter_ydata' > 0 & scatter_ydata' < array_size(2),:);

linind_cropped_scatter_xy_data = sub2ind(array_size,...
    cropped_scatter_xy_data(:,1),cropped_scatter_xy_data(:,2));

figure(42)
plot(cropped_scatter_xy_data(:,1),cropped_scatter_xy_data(:,2),'.')
%
%% surf plot of raw topography.
figure(43)
smoothed_shifted_surf_zdata = ...
    imgaussfilt(shifted_surf_zdata,20);
z_levels = 10:10:70;
h43 = surf(smoothed_shifted_surf_zdata);
hold on
contour3(smoothed_shifted_surf_zdata,z_levels,...
    'LineColor',[0,0,0])
scatter3(cropped_scatter_xy_data(:,1),...
    cropped_scatter_xy_data(:,2),...
    100+ones(size(cropped_scatter_xy_data(:,1))),1,...
    'MarkerFaceColor',[1.0, 0.6, 0],...
    'MarkerEdgeColor','none')

hold off


set(h43,'EdgeColor','none');
view(2)
colormap(gray)
colorbar
axis equal
axis tight
%
%% get z values at the xy location of the points.

scatter_z = shifted_surf_zdata(linind_cropped_scatter_xy_data);

bins_edges = 0.5:1:100;
N_surf_z = histcounts(shifted_surf_zdata,bins_edges,...
    'Normalization', 'probability');
N_scatter_z = histcounts(scatter_z,bins_edges,...
    'Normalization', 'probability');

figure(44)
bar(bins_edges(1:end-1),N_surf_z,1,'FaceAlpha',0.5)
hold on
bar(bins_edges(1:end-1),N_scatter_z,1,'FaceAlpha',0.5)
hold off
%
%% crop4, cropping to 4x4 microns.
size_x4 = 400;
size_y4 = 400;
min_x4 = 1070;
min_y4 = 780;
max_x4 = min_x4 + size_x4;
max_y4 = min_y4 + size_y4;

z_levels4 = 5:5:70;

smoothed_cropped4_z_data = smoothed_shifted_surf_zdata(...
    min_y4:1:max_y4,min_x4:1:max_x4);

cropped4_scatter_xy_data = scatter_xy_data(...
    scatter_xy_data(:,1) > min_x4 & ...
    scatter_xy_data(:,1) < max_x4 &...
    scatter_xy_data(:,2) > min_y4 &...
    scatter_xy_data(:,2) < max_y4,:);

shifted_cropped4_scatter_xy_data = ...
    cropped4_scatter_xy_data - [min_x4,min_y4]; 
% linind_cropped4_scatter_xy_data = sub2ind([size_x4,size_y4],...
%     cropped4_scatter_xy_data(:,1),cropped4_scatter_xy_data(:,2));

figure(44)
h44 = surf(smoothed_cropped4_z_data);
colormap(gray)
set(h44,'EdgeColor','none');
hold on
contour3(smoothed_cropped4_z_data,z_levels4,...
    'LineColor',[0,0,0])
colorbar
scatter3(shifted_cropped4_scatter_xy_data(:,1),...
    shifted_cropped4_scatter_xy_data(:,2),...
    100+ones(size(shifted_cropped4_scatter_xy_data(:,1))),3,...
    'MarkerFaceColor',[1.0, 0.6, 0],...
    'MarkerEdgeColor','none')
hold off

view(2)
axis equal 
axis tight
grid off
caxis([25 70])
%
%% crop2, cropping to 2x2 microns.
size_x2 = 200;
size_y2 = 200;
min_x2 = 1070 + 100;
min_y2 = 780 + 100;
max_x2 = min_x2 + size_x2;
max_y2 = min_y2 + size_y2;


smoothed_cropped2_z_data = smoothed_shifted_surf_zdata(...
    min_y2:1:max_y2,min_x2:1:max_x2);

cropped2_scatter_xy_data = scatter_xy_data(...
    scatter_xy_data(:,1) > min_x2 & ...
    scatter_xy_data(:,1) < max_x2 &...
    scatter_xy_data(:,2) > min_y2 &...
    scatter_xy_data(:,2) < max_y2,:);

shifted_cropped2_scatter_xy_data = ...
    cropped2_scatter_xy_data - [min_x2,min_y2]; 
% linind_cropped4_scatter_xy_data = sub2ind([size_x4,size_y4],...
%     cropped4_scatter_xy_data(:,1),cropped4_scatter_xy_data(:,2));

figure(45)
h45 = surf(smoothed_cropped2_z_data);
colormap(gray)
set(h45,'EdgeColor','none');
hold on
contour3(smoothed_cropped2_z_data,z_levels4,...
    'LineColor',[0,0,0])
colorbar
scatter3(shifted_cropped2_scatter_xy_data(:,1),...
    shifted_cropped2_scatter_xy_data(:,2),...
    100+ones(size(shifted_cropped2_scatter_xy_data(:,1))),6,...
    'MarkerFaceColor',[1.0, 0.6, 0],...
    'MarkerEdgeColor','none')
hold off

view(2)
axis equal 
axis tight
grid off
caxis([30 60])
%
%% get z values at the xy location of the points in 2x2.
linind_cropped_2x2_scatter_xy_data = ...
    sub2ind([size_x2,size_y2],...
    shifted_cropped2_scatter_xy_data(:,1),...
    shifted_cropped2_scatter_xy_data(:,2));

scatter_2x2_z = smoothed_cropped2_z_data(...
    linind_cropped_2x2_scatter_xy_data);

bins_edges = 0.5:1:100;
N_surf_z_2x2 = histcounts(smoothed_cropped2_z_data,bins_edges,...
    'Normalization', 'probability');
N_scatter_z_2x2 = histcounts(scatter_2x2_z,bins_edges,...
    'Normalization', 'probability');

figure(46)
bar(bins_edges(1:end-1),N_surf_z_2x2,1,...
    'FaceColor',0.4*[1 1 1],'FaceAlpha',0.5)
hold on
bar(bins_edges(1:end-1),N_scatter_z_2x2,1,'FaceAlpha',0.5)
hold off



%









