function [] = plotTrainingDataModel3(...
    plots_parameters,data_to_plot)

tcr_x = data_to_plot.tcr.x;
tcr_y = data_to_plot.tcr.y;
tcr_r = data_to_plot.tcr.r;
cd45_x = data_to_plot.cd45.x;
cd45_y = data_to_plot.cd45.y;
ptcr_x = data_to_plot.ptcr.x;
ptcr_y = data_to_plot.ptcr.y;
ptcr_r = data_to_plot.ptcr.r;
decay_length_nm = data_to_plot.decay_length_nm;
depletion_nm = data_to_plot.depletion_nm;
phos_ratio = data_to_plot.phos_ratio;
Rg_ratio = data_to_plot.Rg_ratio;

alck_array = data_to_plot.Lck_array;

tcr_color = plots_parameters.tcr.color;
cd45_color = plots_parameters.cd45.color;
ptcr_color = plots_parameters.ptcr.color;
ms = plots_parameters.ms;

magenta_colormap = plots_parameters.magenta_colormap;

figure(1)
clf
f = 0.2;
suptitle(['Decay length = ',num2str(decay_length_nm),'nm,',...
          ' Depletion = ',num2str(depletion_nm),'nm'])

%% surf Lck*, scatter tcr and cd45
subplot(1,3,1);
if 1 %plot_Lck_array
    hLck = surf(alck_array'/sum(alck_array(:)));
    view(2)
    set(hLck,'EdgeColor','none')
    colormap(magenta_colormap)
    grid off
    box on
    alpha color
    alpha scaled
end

% scatter tcr
if 1 %plot_tcr
    hold on
    scatter3(tcr_x,tcr_y,100+ones(size(tcr_x)),...
        ms,...
        'MarkerEdgeColor','none',...
        'MarkerFaceColor',tcr_color)
    hold off
end

% scatter cd45
if 1 %plot_cd45
    hold on
    scatter3(cd45_x,cd45_y,100+ones(size(cd45_x)),...
        ms,...
        'MarkerEdgeColor','none',...
        'MarkerFaceColor',cd45_color)
    hold off
end

title(['\fontsize{14}'...
       '\color[rgb]{0.1 0.62 0.47}TCR',...
       '\color{black}/',...
       '\color[rgb]{0.75 0.0 0.0}CD45',...
       '\color{black}/',...
       '\color[rgb]{magenta}Lck*'])
grid off
box on
axis equal
axis tight

axis([f*size(alck_array,1), (1-f)*size(alck_array,1),...
      f*size(alck_array,2), (1-f)*size(alck_array,2)])

xticks([-size(alck_array,1):50:size(alck_array,1)])
yticks([-size(alck_array,1):50:size(alck_array,1)])
xticklabels(10*(xticks - 100))
yticklabels(10*(yticks - 100))
%
%% scatter tcr and ptcr %%%%%
subplot(1,3,2);
if 1 %plot_tcr
    hold on
    scatter3(tcr_x,tcr_y,100+ones(size(tcr_x)),...
        ms,...
        'MarkerEdgeColor','none',...
        'MarkerFaceColor',tcr_color)
    hold off
end

if 1 %plot_ptcr
    hold on
    scatter3(ptcr_x,ptcr_y,100+ones(size(ptcr_x)),...
        1.5*ms,...
        'MarkerEdgeColor','none',...
        'MarkerFaceColor',ptcr_color)
    hold off
end

title(['\fontsize{14}'...
       '\color[rgb]{0.1 0.62 0.47}TCR',...
       '\color{black}/',...
       '\color[rgb]{0.9 0.67 0.0}pTCR'])
grid off
box on
axis equal
axis tight

axis([f*size(alck_array,1), (1-f)*size(alck_array,1),...
      f*size(alck_array,2), (1-f)*size(alck_array,2)])

xticks([-size(alck_array,1):50:size(alck_array,1)])
yticks([-size(alck_array,1):50:size(alck_array,1)])
xticklabels(10*(xticks - 100))
yticklabels(10*(yticks - 100))
%
%% Make histograms
edges = 0:1:60;
[tcr_counts,edges] = histcounts(tcr_r,edges);
[ptcr_counts,edges] = histcounts(ptcr_r,edges);

r_bins = edges(1:end-1);
normalyzed_tcr_counts = tcr_counts./r_bins;
normalyzed_ptcr_counts = ptcr_counts./r_bins;

mirrored_normalyzed_tcr_counts = ...
    [fliplr(normalyzed_tcr_counts(2:end)),...
     normalyzed_tcr_counts];

mirrored_r_bins = -59:1:59;
%
%%

smooth_range = 7;
smoothed_normalyzed_tcr_counts = ...
    smooth(normalyzed_tcr_counts,smooth_range);

smoothed_normalyzed_ptcr_counts = ...
    smooth(normalyzed_ptcr_counts,smooth_range);

subplot(1,3,3); %h3 = 
% right tcr
hold on
hbar_tcr = bar(r_bins,smoothed_normalyzed_tcr_counts,1);
hbar_tcr.EdgeColor = 'none';
hbar_tcr.FaceColor = tcr_color;
hold off

% right ptcr
hold on
hbar_ptcr = bar(r_bins,smoothed_normalyzed_ptcr_counts,1);
hbar_ptcr.EdgeColor = 'none';
hbar_ptcr.FaceColor = ptcr_color;
hold off

% left tcr
hold on
hbar_tcr = bar(-r_bins,smoothed_normalyzed_tcr_counts,1);
hbar_tcr.EdgeColor = 'none';
hbar_tcr.FaceColor = tcr_color;
hold off

% left ptcr
hold on
hbar_ptcr = bar(-r_bins,smoothed_normalyzed_ptcr_counts,1);
hbar_ptcr.EdgeColor = 'none';
hbar_ptcr.FaceColor = ptcr_color;
hold off

max_y = max(smoothed_normalyzed_tcr_counts);
hold on
plot([0, 0],[0, 1],'k-.','LineWidth',0.25)
title(['\fontsize{14}'...
       '\color[rgb]{0.1 0.62 0.47}TCR',...
       '\color{black}/',...
       '\color[rgb]{0.9 0.67 0.0}pTCR'])   
hold off
grid off
box on
axis square
axis tight

% axis([f*size(alck_array,1), (1-f)*size(alck_array,1),...
%       f*size(alck_array,2), (1-f)*size(alck_array,2)])
% % caxis([0 70])
xticks([-size(alck_array,1):50:size(alck_array,1)])
% yticks([0:50:size(alck_array,2)])
xticklabels(10*xticks)
% yticklabels(10*yticks)
%
%%
pause(0.1)
drawnow
end