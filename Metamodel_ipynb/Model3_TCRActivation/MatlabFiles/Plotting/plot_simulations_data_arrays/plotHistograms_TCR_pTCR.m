function [] = plotHistograms_TCR_pTCR(...
    plots_parameters,Z1,...
    tcr_x,tcr_y,cd45_x,cd45_y,ptcr_x,ptcr_y,alpha)

%% get radius of gyration
% N_ratio = length(ptcr_x)/length(tcr_x);
% [r_tcr,~] = radiusOfGyration(tcr_x,tcr_y);
% [r_ptcr,~] = radiusOfGyration(ptcr_x,ptcr_y);
% Rg_ratio = Rg_ptcr/Rg_tcr;

%%%
mean_tcr_x = mean(tcr_x);
mean_tcr_y = mean(tcr_y);

r_tcr_squared = ((tcr_x - mean_tcr_x).^2 +...
                 (tcr_y - mean_tcr_y).^2);
r_tcr = sqrt(r_tcr_squared);

r_ptcr_squared = ((ptcr_x - mean_tcr_x).^2 +...
                  (ptcr_y - mean_tcr_y).^2);
r_ptcr = sqrt(r_ptcr_squared);
%%%

edges = 0:1:60;
[tcr_counts,edges] = histcounts(r_tcr,edges);
[ptcr_counts,edges] = histcounts(r_ptcr,edges);

r_bin = 1:length(edges)-1;
normalyzed_tcr_counts = tcr_counts./r_bin;
normalyzed_ptcr_counts = ptcr_counts./r_bin;

%
%%
smooth_range = 7;
smoothed_normalyzed_tcr_counts = ...
    smooth(normalyzed_tcr_counts,smooth_range);

smoothed_normalyzed_ptcr_counts = ...
    smooth(normalyzed_ptcr_counts,smooth_range);

% right tcr
hold on
hbar_tcr = bar(r_bin,smoothed_normalyzed_tcr_counts,1);
hbar_tcr.EdgeColor = 'none';
hbar_tcr.FaceColor = plots_parameters.tcr.color;
hold off

% right ptcr
hold on
hbar_ptcr = bar(r_bin,smoothed_normalyzed_ptcr_counts,1);
hbar_ptcr.EdgeColor = 'none';
hbar_ptcr.FaceColor = plots_parameters.ptcr.color;
hold off

% left tcr
hold on
hbar2_tcr = bar(-r_bin,smoothed_normalyzed_tcr_counts,1);
hbar2_tcr.EdgeColor = 'none';
hbar2_tcr.FaceColor = plots_parameters.tcr.color;
hold off

% left ptcr
hold on
hbar2_ptcr = bar(-r_bin,smoothed_normalyzed_ptcr_counts,1);
hbar2_ptcr.EdgeColor = 'none';
hbar2_ptcr.FaceColor = plots_parameters.ptcr.color;
hold off

max_y = 1; %max(smoothed_normalyzed_tcr_counts);

hold on
plot([0, 0],[0, max_y],'k-.','LineWidth',0.25)
hold off

%%%
max_x = 60;
% max_y = 1;

axis([-max_x, max_x, 0, max_y])

xticks([])
yticks([])
box on
drawnow

end