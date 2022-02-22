% model3_plot_Array_Parameters


%% sizes: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subplots and gaps sizes (relative to the figure size):
parameters.plots.N_cols = 5;
parameters.plots.N_rows = 6;
parameters.plots.gapx0 = 0.12; % relative initial gap on the left.
parameters.plots.gapy0 = 0.125; % relative initial gap on the bottom.
parameters.plots.gapx = 0.02; % relative gap x between subplots.
parameters.plots.gapy = 0.02; % relative gap x between subplots.
parameters.plots.size_x = 0.75/parameters.plots.N_cols; % relative subplots x size.
parameters.plots.size_y = 0.75/parameters.plots.N_rows; % relative subplots y size.

parameters.plots.lim1 = 60; % pixels
parameters.plots.tick1 = 50; % pixels
parameters.plots.lim2 = 30; % pixels
parameters.plots.tick2 = 25; % pixels

parameters.plots.axis_off = 1;
parameters.plots.depletions = [-250,0:10:200];
parameters.plots.decayLengths = 10:10:200;

% selected indices to plot as subplots:
parameters.plots.s_dep_ind = [1,2:5:22]; % selected, N = N_rows
parameters.plots.s_dec_ind = [2,5,10,15,20]; % selected, N = N_cols

%% arrays: %%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.arrays.pixel_size_nm = 10; % nm
parameters.arrays.mic2nm = 1000;  % microns to nm.
%
parameters.arrays.size_x_microns = 2;
parameters.arrays.size_y_microns = 2;
%
%% TCR_locations %%%%%%%%%%%%%%%%%%%%%%
parameters.TCR.color = [0.0 0.6 0.0];
parameters.TCR.cluster_density = 1000; % #/micron^2
parameters.TCR.r1_microns = 0;
parameters.TCR.r2_microns = 0.25; % microns

parameters.CD45.color = [1.0 0.0 0.0];
parameters.CD45.cluster_density = 1000; % #/micron^2
parameters.CD45.decay_length_nm = 10; 
parameters.CD45.width_microns = 0.3;
parameters.CD45.width_microns2 = 0.55;

parameters.pTCR.color = [1.0 0.5 0.0]; %[1.0 0.0 1.0];

parameters.line_color = 0.2*[1.0 1.0 1.0];
%
%% colormaps: %%%%%%%%%%%%%%%%%%%%%%%%%
Nc = 64;
parameters.colormaps.orange_blue = orangeBlueColormap(Nc);
parameters.colormaps.orange_gray = orangeGrayColormap(Nc);
parameters.colormaps.orange_fixed = orangeFixedColormap(Nc);
parameters.colormaps.magenta_fixed = magentaFixedColormap(Nc);
%









