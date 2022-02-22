function plotModel1Data(plot_what,plot_parameters)

%% get simulation data
% all_linind_cell1 = simulation_data.Cells(1).molecules(:,2);
% linind_tcr = all_linind_cell1(simulation_data.Cells(1).molecules(:,3) == 1); 
% linind_cd45 = all_linind_cell1(simulation_data.Cells(1).molecules(:,3) == 3); 
% 
% Z1 = simulation_data.Cells(1).Z;
% 
% [tcr_x,tcr_y] = ind2sub(size(Z1),linind_tcr);
% [cd45_x,cd45_y] = ind2sub(size(Z1),linind_cd45);

%% get plot parameters

tcr_color = plot_parameters.tcr_color;
cd45_color = plot_parameters.cd45_color;
ptcr_color = plot_parameters.ptcr_color;

N_colors = plot_parameters.N_colors;
                
magenta_colormap = plot_parameters.magenta_colormap;              
orange_colormap = plot_parameters.orange_colormap;
                
fig_position = plot_parameters.fig_position;
fig_position_shift_x = plot_parameters.fig_position_shift_x;
fig_position_shift_y = plot_parameters.fig_position_shift_y;


%% plot what
if ismember('raw_membrane',plot_what)
    
end


end