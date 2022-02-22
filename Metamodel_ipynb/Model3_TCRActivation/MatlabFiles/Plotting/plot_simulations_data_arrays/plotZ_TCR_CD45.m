function [] = plotZ_TCR_CD45(plots_parameters,Z1,...
    tcr_x,tcr_y,cd45_x,cd45_y,alpha)

smoothed_Z = imgaussfilt(Z1,8);
contours_levels = 10:10:70;

hz = surf(smoothed_Z');
hold on
hc = contour(smoothed_Z');
[M,c] = contour3(smoothed_Z', contours_levels);
c.Color = [0.0 0.0 0.0];
hold off
colormap(gray)
hz.EdgeColor = 'none';

% hold on
% scatter3(tcr_x,tcr_y,100+ones(size(tcr_x)),...
%     plots_parameters.marker_size,...
%     'MarkerEdgeColor','none',...
%     'MarkerFaceColor',plots_parameters.tcr.color,...
%     'MarkerFaceAlpha',1.0)
% 
% scatter3(cd45_x,cd45_y,100+ones(size(cd45_x)),...
%     plots_parameters.marker_size,...
%     'MarkerEdgeColor','none',...
%     'MarkerFaceColor',plots_parameters.cd45.color,...
%     'MarkerFaceAlpha',alpha)
% 
% hold off
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
caxis([10 50])
xticks([])
yticks([])
% xticklabels(10*xticks)
% yticklabels(10*yticks)
% colorbar
drawnow

end