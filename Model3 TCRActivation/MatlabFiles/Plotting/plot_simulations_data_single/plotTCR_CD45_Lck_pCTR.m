function [] = plotTCR_CD45_Lck_pCTR(plots_parameters,...
    tcr_x,tcr_y,cd45_x,cd45_y,ptcr_x,ptcr_y,...
    Z1,Lck_phos_array)

hLck = surf(Lck_phos_array'/max(Lck_phos_array(:)));
set(hLck,'EdgeColor','none')
view(2)
colormap(plots_parameters.colormaps.magenta_fixed)
grid off
box on
alpha color
alpha scaled

hold on
scatter3(tcr_x,tcr_y,100*ones(size(tcr_x)),...
    plots_parameters.marker_size,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',plots_parameters.TCR.color)

scatter3(cd45_x,cd45_y,100*ones(size(cd45_x)),...
    plots_parameters.marker_size,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',plots_parameters.CD45.color)

scatter3(ptcr_x,ptcr_y,100*ones(size(ptcr_x)),...
    plots_parameters.marker_size*1.5,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',plots_parameters.pTCR.color)
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