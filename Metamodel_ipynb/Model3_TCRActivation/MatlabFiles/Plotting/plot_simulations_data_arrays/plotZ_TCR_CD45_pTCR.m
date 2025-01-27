function [] = plotZ_TCR_CD45_pTCR(plots_parameters,Z1,...
    tcr_x,tcr_y,cd45_x,cd45_y,ptcr_x,ptcr_y,alpha)

smoothed_Z = imgaussfilt(Z1,8);

hz = surf(smoothed_Z');
colormap(gray)
hz.EdgeColor = 'none';

hold on
scatter3(tcr_x,tcr_y,100+ones(size(tcr_x)),...
    plots_parameters.marker_size,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',plots_parameters.TCR.color,...
    'MarkerFaceAlpha',1.0)

scatter3(cd45_x,cd45_y,100+ones(size(cd45_x)),...
    plots_parameters.marker_size,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',plots_parameters.CD45.color,...
    'MarkerFaceAlpha',alpha)

scatter3(ptcr_x,ptcr_y,100+ones(size(ptcr_x)),...
    plots_parameters.marker_size,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',plots_parameters.pTCR.color,...
    'MarkerFaceAlpha',1.0)

hold off
grid off
box on
view(2)
axis equal
axis tight
f = 0.2;
axis([f*size(Z1,1),...
      (1-f)*size(Z1,1),...
      f*size(Z1,2),...
      (1-f)*size(Z1,2)])

xticks([])
yticks([])
% xticklabels(10*xticks)
% yticklabels(10*yticks)
% colorbar
drawnow

end