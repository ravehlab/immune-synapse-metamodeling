function [] = plotWhat(plot_parameters,...
    data_to_plot,plot_what)

%% get plot_parameters:
tcr_color = plot_parameters.tcr.color;
cd45_color = plot_parameters.cd45.color;
ptcr_color = plot_parameters.ptcr.color;

N_colors = plot_parameters.N_colors;
magenta_colormap = plot_parameters.magenta_colormap;
orange_colormap = plot_parameters.orange_colormap;

fig_position = plot_parameters.fig_position;
fig_position_shift_x = plot_parameters.fig_position_shift_x;
fig_position_shift_y = plot_parameters.fig_position_shift_y;

ms = plot_parameters.ms;
contours_levels = plot_parameters.contours_levels;
%
%% get plot_data:
Z1 = data_to_plot.membrane_z;
smoothed_Z = data_to_plot.smoothed_Z;
Lck_array = data_to_plot.Lck_array;
Lck_array = Lck_array/max(Lck_array(:));
tcr_x = data_to_plot.tcr.x;
tcr_y = data_to_plot.tcr.y;
cd45_x = data_to_plot.cd45.x;
cd45_y = data_to_plot.cd45.y;
ptcr_x = data_to_plot.ptcr.x;
ptcr_y = data_to_plot.ptcr.y;
%
%% get plot_what:
% plot_colorbar = plot_what.plot_colorbar;

% plot surface
if plot_what.plot_surface
    hold on
    hfig = surf(Z1');
    hfig.EdgeColor = 'none';
    colormap(gray)
    view(2)
    if plot_colorbar
       colorbar 
    end
    hold off
end

% plot smoothed surface
if plot_what.plot_smoothed_surface
    hold on
    hfig = surf(smoothed_Z');
    hfig.EdgeColor = 'none';
    colormap(gray)
    view(2)
    if plot_what.plot_colorbar
       colorbar 
    end
    hold off
end

% plot contours Z
if plot_what.plot_contours
    hold on
    [M,c] = contour3(smoothed_Z', contours_levels);
    c.Color = [0.0 0.0 0.0];
    hold off
end

% plot Lck array
if plot_what.plot_Lck_array
    hLck = surf(Lck_array');

    set(hLck,'EdgeColor','none')
    colormap(magenta_colormap)
    grid off
    box on
    alpha color
    alpha scaled
end

% scatter tcr
if plot_what.plot_tcr
    hold on
    scatter3(tcr_x,tcr_y,100+ones(size(tcr_x)),...
        ms,...
        'MarkerEdgeColor','none',...
        'MarkerFaceColor',tcr_color)
    hold off
end

% scatter cd45
if plot_what.plot_cd45
    hold on
    scatter3(cd45_x,cd45_y,100+ones(size(cd45_x)),...
        ms,...
        'MarkerEdgeColor','none',...
        'MarkerFaceColor',cd45_color)
    hold off
end

% scatter cd45
if plot_what.plot_ptcr
    hold on
    scatter3(ptcr_x,ptcr_y,100+ones(size(ptcr_x)),...
        1.5*ms,...
        'MarkerEdgeColor','none',...
        'MarkerFaceColor',ptcr_color)
    hold off
end

grid off
box on
view(2)
axis equal
axis tight
f = 0.2;
axis([f*size(Z1,1), (1-f)*size(Z1,1),...
      f*size(Z1,2), (1-f)*size(Z1,2)])
caxis([0 70])
xticks([0:100:size(Z1,1)])
yticks([0:100:size(Z1,2)])
xticklabels(10*xticks)
yticklabels(10*yticks)

end