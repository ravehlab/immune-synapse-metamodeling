% get_simulation_results_model1
clear
clc
path1 = 'C:\Users\yairn\Downloads\results_20211026152628-20211027T133406Z-001\results_20211026152628';
addpath(path1);
load('RESULTS_OF_RUNS')
%%
t = 101;
sim_data = results_of_run(t);
cell1_data = sim_data{1,1}.Cells(1);
Z = cell1_data.Z;
molecules = cell1_data.molecules;

% molecules properties
molecules_linind = molecules(:,2);
molecules_type = molecules(:,3);
molecules_z = molecules(:,4);

tcr_linind = molecules_linind(molecules_type == 1);
cd45_linind = molecules_linind(molecules_type == 3);

[tcr_x,tcr_y] = ind2sub(array_size,tcr_linind);
[cd45_x,cd45_y] = ind2sub(array_size,cd45_linind);

tcr_color = [0.0, 0.6, 0.0];
cd45_color = [1.0, 0.0, 0.0];

%% plot results
array_size = size(Z);
% Iblur2 = imgaussfilt(I,4);
Z_smoothed = imgaussfilt(Z,4);
levels = 0:10:70;
if 0
    figure(1)
    h1 = surf(Z);
    set(h1,'EdgeColor','none')
    view(2)
    axis equal
    axis tight
    axis([1 array_size(1) 1 array_size(2)])
end
figure(2)
h2 = surf(Z_smoothed);
set(h2,'EdgeColor','none')
hold on
contour3(Z_smoothed,levels,'k')
hold off
view(2)
axis equal
axis tight
axis([1 array_size(1) 1 array_size(2)])
caxis([0 70])
colormap(gray)
colorbar

hold on
plot3(tcr_x,tcr_y,100*ones(size(tcr_x)),'.','Color',tcr_color)
plot3(cd45_x,cd45_y,100*ones(size(cd45_x)),'.','Color',cd45_color)
hold off






