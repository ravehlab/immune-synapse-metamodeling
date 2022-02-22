function plots_parameters = setPlotsParameters()

%% set colors
tcr_color = [0.1 0.62 0.47]; %[0.0, 0.85, 0.0];
cd45_color = [0.75 0.0 0.0]; %[1.0, 0.0, 0.0]; %
ptcr_color = [0.9 0.67 0.0]; %[1.0, 0.5, 0.0];
lck_color = [1.0, 0.0, 1.0];
%
N_colors = 64;
                
magenta_colormap = ([ones(N_colors,1),...
                     zeros(N_colors,1),...
                     ones(N_colors,1)]);
                 
orange_colormap = ([ones(N_colors,1),...
                    0.75*ones(N_colors,1),...
                    zeros(N_colors,1)]);
                
fig_position = [300,200,400,400];
fig_position_shift_x = 50;
fig_position_shift_y = 50;

%% create 'plots_parameters' struct:
plots_parameters.tcr.color = tcr_color;
plots_parameters.cd45.color = cd45_color;
plots_parameters.ptcr.color = ptcr_color;

plots_parameters.N_colors = N_colors;
plots_parameters.magenta_colormap = magenta_colormap;
plots_parameters.orange_colormap = orange_colormap;

plots_parameters.fig_position = fig_position;
plots_parameters.fig_position_shift_x = fig_position_shift_x;
plots_parameters.fig_position_shift_y = fig_position_shift_y;

plots_parameters.ms = 10; %5
plots_parameters.contours_levels = 10:10:70;

end