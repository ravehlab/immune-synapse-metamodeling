% results_of_runs_Barak_experimental_arrays
clear
clc

%%
% model3_plot_Array_Parameters
%% sizes: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plots_parameters.fig_position = [300,50,700,700];
plots_parameters.fig_position_shift_x = 50;
plots_parameters.fig_position_shift_y = 50;

% subplots and gaps sizes (relative to the figure size):
plots_parameters.N_cols = 6;
plots_parameters.N_rows = 6;
plots_parameters.gapx0 = 0.06; % relative initial gap on the left.
plots_parameters.gapy0 = 0.06; % relative initial gap on the bottom.
plots_parameters.gapx = 0.02; % relative gap x between subplots.
plots_parameters.gapy = 0.02; % relative gap x between subplots.
plots_parameters.size_x = 0.75/plots_parameters.N_cols; % relative subplots x size.
plots_parameters.size_y = 0.75/plots_parameters.N_rows; % relative subplots y size.

plots_parameters.lim1 = 60; % pixels
plots_parameters.tick1 = 50; % pixels
plots_parameters.lim2 = 30; % pixels
plots_parameters.tick2 = 25; % pixels

plots_parameters.axis_off = 1;

%% arrays: %%%%%%%%%%%%%%%%%%%%%%%%%%%%
plots_parameters.arrays.pixel_size_nm = 10; % nm
plots_parameters.arrays.mic2nm = 1000;  % microns to nm.
%
plots_parameters.arrays.size_x_microns = 2;
plots_parameters.arrays.size_y_microns = 2;
%
%% TCR_locations %%%%%%%%%%%%%%%%%%%%%%
plots_parameters.tcr.color = [0.1 0.62 0.47]; %[0.0 0.85 0.0];
plots_parameters.tcr.cluster_density = 1000; % #/micron^2

plots_parameters.cd45.color = [0.75 0.0 0.0];%[217,95,2]/255;%[117,112,179]/255; %[1.0 0.0 0.0];
plots_parameters.cd45.decay_length_nm = 10;

plots_parameters.ptcr.color = [0.9 0.67 0.0]; %[1.0 0.6 0.0]; %[1.0 0.0 1.0];

plots_parameters.line_color = 0.2*[1.0 1.0 1.0];

% colormaps: %%%%%%%%%%%%%%%%%%%%%%%%%
plots_parameters.colormaps.N_colors = 64;

magenta_colormap = ([ones(plots_parameters.colormaps.N_colors,1),...
    zeros(plots_parameters.colormaps.N_colors,1),...
    ones(plots_parameters.colormaps.N_colors,1)]);

orange_colormap = ([ones(plots_parameters.colormaps.N_colors,1),...
    0.75*ones(plots_parameters.colormaps.N_colors,1),...
    zeros(plots_parameters.colormaps.N_colors,1)]);
%

plots_parameters.colormaps.orange_fixed = orange_colormap;
plots_parameters.colormaps.magenta_fixed = magenta_colormap;

plots_parameters.marker_size = 3;

% fig51 = figure(51);
% set(fig51,'Position',plots_parameters.fig_position)

fig52 = figure(52);
set(fig52,'Position',plots_parameters.fig_position)

%

% load('C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\results_20211020133341\RESULTS_OF_RUNS.mat');
% load('C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\results_20211020185919\RESULTS_OF_RUNS.mat');
% load('C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\results_20211021132010\RESULTS_OF_RUNS.mat');
% load('C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\results_20211021132946\RESULTS_OF_RUNS.mat');
% load('C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\results_20211021134315\RESULTS_OF_RUNS.mat');
% load('C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\results_20211021134959\RESULTS_OF_RUNS.mat');
% load('C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\results_20211021141456\RESULTS_OF_RUNS.mat');

%%% plotting from bottom to top
% dict:
% 1, g:overlap
% 2, a:5
% 3, f:12.5
% 4, e:25
% 5, b:50
% 6, c:100
decay_lengths_nm = [20, 50, 100, 150, 200, 5000];
% letters = ["g","a","f","e","b","c"];
letters = ["F","A","B","C","D","E"];

plot_tcr_cd45 = 0;
plot_z_tcr_cd45 = 1;
plot_lck_phos_array_tcr_cd45 = 0;
plot_tcr_cd45_ptcr = 0;
plotZ_tcr_cd45_ptcr = 0;
bar_Z_tcr_cd45_ptcr_hist = 0;
plot_cross_section_Z_tcr_cd45_ptcr = 0;
plot_histograms_tcr_ptcr = 0;

N_ratio_array = zeros(length(letters),plots_parameters.N_cols);
Rg_ratio_array = zeros(length(letters),plots_parameters.N_cols);

for row = 1:length(letters)
    for col = 1:plots_parameters.N_cols
        decay_length_nm = decay_lengths_nm(col);
        off_disk_radius_nm = 15;
        
        letter = letters(row);
        
        kstr = letter;
        if kstr == 'a'
            k = 5;
            path1 = 'C:\Users\yairn\Downloads\results_20211026152628-20211027T133406Z-001\results_20211026152628';
        elseif kstr == 'b'
            k = 50;
            path1 = 'C:\Users\yairn\Downloads\results_20211026180327-20211027T132524Z-001\results_20211026180327';
        elseif kstr == 'c'
            k = 100;
            path1 = 'C:\Users\yairn\Downloads\results_20211026174624-20211027T132656Z-001\results_20211026174624';
        elseif kstr == 'd'
            
            path1 = 'C:\Users\yairn\Downloads\results_20211026173646-20211027T132825Z-001\results_20211026173646';
        elseif kstr == 'e'
            k = 25;
            path1 = 'C:\Users\yairn\Downloads\results_20211026173008-20211027T132951Z-001\results_20211026173008';
        elseif kstr == 'f'
            k = 12.5;
            path1 = 'C:\Users\yairn\Downloads\results_20211026154442-20211027T133110Z-001\results_20211026154442';
        elseif kstr == 'g'
            k = 25; % overlap
            path1 = 'C:\Users\yairn\Downloads\results_20211026181617-20211027T132352Z-001\results_20211026181617';
        
        
        % double size tcr:
        elseif kstr == 'A'
            k = 5;
            path1 = 'C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\20211205\results_20211205194022_k5';
        elseif kstr == 'B'
            k = 12.5;
            path1 = 'C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\20211205\results_20211205195843_k12_5';
        elseif kstr == 'C'
            k = 25;
            path1 = 'C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\20211205\results_20211205200708_k25';
        elseif kstr == 'D'
            k = 50;
            path1 = 'C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\20211205\results_20211205201620_k50';
        elseif kstr == 'E'
            k = 100;
            path1 = 'C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\20211205\results_20211205202622_k100';
        elseif kstr == 'F'
            k = 25; % overlap
            path1 = 'C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\20211205\results_20211205203902_k25_overlap';
        end        
        
        addpath(path1);
        load('RESULTS_OF_RUNS.mat')
        
        %% load simulation data
        
        %% for overlapping distribution get the initial conditions
        if kstr == 'F'
            t = 1;
            all_linind_cell1_t = results_of_run{t,1}.Cells(1).molecules(:,2);
            linind_tcr = all_linind_cell1_t(simulation_data.Cells(1).molecules(:,3) == 1);
            linind_cd45 = all_linind_cell1_t(simulation_data.Cells(1).molecules(:,3) == 3);
            Z1 = results_of_run{t,1}.Cells(1).Z;
        else
            t = 101;
            all_linind_cell1_t = results_of_run{t,1}.Cells(1).molecules(:,2);            
            linind_tcr = all_linind_cell1_t(simulation_data.Cells(1).molecules(:,3) == 1);
            linind_cd45 = all_linind_cell1_t(simulation_data.Cells(1).molecules(:,3) == 3);
            Z1 = results_of_run{t,1}.Cells(1).Z;
        end
        rmpath(path1);
        %%
%         linind_tcr = 1;
%         linind_cd45 = 19900;
        %%
        [tcr_x,tcr_y] = ind2sub(size(Z1),linind_tcr);
        [cd45_x,cd45_y] = ind2sub(size(Z1),linind_cd45);
        
        %%% new
        %% get molecules' r relative to center and molecules' z:
        [tcr_r,tcr_z] = xyZ2rz(tcr_x,tcr_y,Z1);
        [cd45_r,cd45_z] = xyZ2rz(cd45_x,cd45_y,Z1);
        %%
%         plotTCR_CD45_Z_2D(tcr_r,tcr_z,cd45_r,cd45_z,Z1)
        %%%
        %% Lck decay length
        
        cd45_array = zeros(size(Z1));
        cd45_array(linind_cd45) = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% active Lck distribution kernel around a single cd45:
        exponential_decay_kernel = ...
            ExponentialDecayKernel(100,...
            plots_parameters.arrays.pixel_size_nm,...
            0);
        %%%
        %% active Lck distribution array:
        lck_phos_array = lckPhosArray(linind_cd45,Z1,...
            decay_length_nm,off_disk_radius_nm,...
            plots_parameters.arrays.pixel_size_nm);
        %%%
        %% phosphorylated tcrs as a function of lck_phos_array:
        phos_threshold = 0.15;
        [ptcr_x,ptcr_y] = ...
            getPhosTCR(linind_tcr,lck_phos_array,phos_threshold);   
 
        %%%
        pos1 = getSubplotPosition(plots_parameters,col,row);
        %% 1 plot tcr and cd45:
        if plot_tcr_cd45
            subplot('Position',pos1)
            plotTCR_CD45(plots_parameters,tcr_x,tcr_y,cd45_x,cd45_y,Z1)
        end
        %%%
        %% 1.5 plot Z, tcr and cd45:
        if plot_z_tcr_cd45
            alpha = 1.0;
            subplot('Position',pos1)
            plotZ_TCR_CD45(plots_parameters,Z1,tcr_x,tcr_y,cd45_x,cd45_y,alpha)
        end
        %%%
        %% 2 plot lck_phos_array, tcr, cd45:
        if plot_lck_phos_array_tcr_cd45
            subplot('Position',pos1)
            plotTCR_CD45_Lck(plots_parameters,...
                tcr_x,tcr_y,cd45_x,cd45_y,Z1,lck_phos_array)
        end
        %%%
        %% 3 plot tcr, cd45, ptcr:
        if plot_tcr_cd45_ptcr
            alpha = 1.0;
            subplot('Position',pos1)
            plotTCR_CD45_pTCR(plots_parameters,Z1,...
                tcr_x,tcr_y,cd45_x,cd45_y,ptcr_x,ptcr_y,alpha)
        end
        %%%
        %% 4 plot Z, tcr, cd45, ptcr:
        if plotZ_tcr_cd45_ptcr
            alpha = 1.0;
            subplot('Position',pos1)
            plotZ_TCR_CD45_pTCR(plots_parameters,Z1,...
                tcr_x,tcr_y,cd45_x,cd45_y,ptcr_x,ptcr_y,alpha)
        end
        %% 5 plot bar Z, tcr, cd45, ptcr:
        if bar_Z_tcr_cd45_ptcr_hist
            alpha = 1.0;
            subplot('Position',pos1)
            plotBarZ_TCR_CD45_pTCR(plots_parameters,Z1,...
                tcr_x,tcr_y,cd45_x,cd45_y,ptcr_x,ptcr_y,alpha)
        end
        
        %% 6 plot cross section Z, tcr, cd45, ptcr:
        if plot_cross_section_Z_tcr_cd45_ptcr
            alpha = 1.0;
            subplot('Position',pos1)
            plotCross_sections_Z_TCR_CD45_pTCR(plots_parameters,Z1,...
                tcr_x,tcr_y,cd45_x,cd45_y,ptcr_x,ptcr_y,alpha)
        end
        
        %% 7 plot histograms, tcr, ptcr:
        if plot_histograms_tcr_ptcr
            alpha = 1.0;
            subplot('Position',pos1)
            plotHistograms_TCR_pTCR(plots_parameters,Z1,...
                tcr_x,tcr_y,cd45_x,cd45_y,ptcr_x,ptcr_y,alpha)
            
            N_ratio = length(ptcr_x)/length(tcr_x);
            [r_tcr,Rg_tcr] = radiusOfGyration(tcr_x,tcr_y);
            [r_ptcr,Rg_ptcr] = radiusOfGyration(ptcr_x,ptcr_y);
            Rg_ratio = Rg_ptcr/Rg_tcr;

            N_ratio_array(row,col) = N_ratio;
            Rg_ratio_array(row,col) = Rg_ratio;
        end
    end % col
    rmpath(path1)
end % row

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
%% 2 plot smoothed topography and contours

hsize = 51;
sigma_smooth = 7;
h = fspecial('gaussian',hsize,sigma_smooth);
smoothed_Z = imfilter(Z1',h,'replicate');

fig12 = figure(12);
set(fig12,'Position',fig_position + ...
    1*[fig_position_shift_x, fig_position_shift_x, 0, 0])
h12 = surf(smoothed_Z);
set(h12, 'EdgeColor','none')
colormap(gray)
% colorbar

hold on
levels = 0:10:70;
[M,c] = contour3(smoothed_Z, levels);
c.Color = [0.0 0.0 0.0];
try
%     scatter3(tcr_x,tcr_y,100*ones(size(tcr_x)),...
%     ms,...
%     'MarkerEdgeColor','none',...
%     'MarkerFaceColor',tcr_color)
%
%     scatter3(tcr_binary_phos_x,tcr_binary_phos_y,100*ones(size(tcr_binary_phos_x)),...
%         ms,...
%         'MarkerEdgeColor','none',...
%         'MarkerFaceColor',ptcr_color)
catch
    
end
hold off

caxis([0 70])
view(2)
grid off
box on
axis equal
axis tight
axis([0 size(Z1,1) 0 size(Z1,2) 0 100])
caxis([0 70])
xticks([0:100:size(Z1,1)])
yticks([0:100:size(Z1,2)])
xticklabels(10*xticks)
yticklabels(10*yticks)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3 plot tcr, cd45, Lck distribution, phosphorylated tcr
tcr_array = zeros(size(Z1));
tcr_array(linind_tcr) = 1;
% tcr_gradual_phos = tcr_array.*phos_ability_array'/...
%     max(max(phos_ability_array));


% tcr_binary_phos = binary_phos_ability_array.*tcr_array;
tcr_binary_phos = rand_norm_phos_ability_array.*tcr_array;


[tcr_binary_phos_x,tcr_binary_phos_y] = ...
    find(tcr_binary_phos);

%%%

fig13 = figure(13);
set(fig13,'Position',fig_position + ...
    2*[fig_position_shift_x, fig_position_shift_x, 0, 0])
hLck = surf(0.01*Lck_array');
% hLck = surf(phos_ability_array');
% hold on
% hLck = contour3(phos_on_array');
% hold off
set(hLck,'EdgeColor','none')
view(2)
colormap(magenta_colormap)
grid off
box on
alpha color
alpha scaled


hold on
scatter3(tcr_x,tcr_y,100*ones(size(tcr_x)),...
    ms,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',tcr_color)
scatter3(cd45_x,cd45_y,100*ones(size(cd45_x)),...
    ms,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',cd45_color)

% scatter3(tcr_binary_phos_x,tcr_binary_phos_y,100*ones(size(tcr_binary_phos_x)),...
%     ms,...
%     'MarkerEdgeColor','none',...
%     'MarkerFaceColor',ptcr_color)

hold off



caxis([0 70])
view(2)
box on
axis equal
axis tight
axis([0 size(Z1,1) 0 size(Z1,2)])
caxis([0 70])
xticks([0:100:size(Z1,1)])
yticks([0:100:size(Z1,2)])
xticklabels(10*xticks)
yticklabels(10*yticks)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot smoothed Z and contours
fig14 = figure(14);
set(fig14,'Position',fig_position + ...
    1*[fig_position_shift_x, fig_position_shift_x, 0, 0])
h14 = surf(smoothed_Z);
set(h14, 'EdgeColor','none')
colormap(gray)
colorbar

hold on
levels = 0:10:70;
[M,c] = contour3(smoothed_Z, levels);
c.Color = [0.0 0.0 0.0];
scatter3(tcr_binary_phos_x,tcr_binary_phos_y,100*ones(size(tcr_binary_phos_x)),...
    ms,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',ptcr_color)


scatter3(tcr_x,tcr_y,100*ones(size(tcr_x)),...
    ms,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',tcr_color)
scatter3(cd45_x,cd45_y,100*ones(size(cd45_x)),...
    ms,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',cd45_color)

hold off

caxis([0 70])
view(2)
box on
axis equal
axis tight
axis([0 size(Z1,1) 0 size(Z1,2)])
caxis([0 70])
xticks([0:100:size(Z1,1)])
yticks([0:100:size(Z1,2)])
xticklabels(10*xticks)
yticklabels(10*yticks)
%% tcr and ptcr
fig15 = figure(15);
set(fig15,'Position',fig_position + ...
    3*[fig_position_shift_x, fig_position_shift_x, 0, 0])

scatter3(tcr_x,tcr_y,100*ones(size(tcr_x)),...
    ms,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',tcr_color,...
    'MarkerFaceAlpha',0.2)

hold on
scatter3(cd45_x,cd45_y,100*ones(size(cd45_x)),...
    ms,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',cd45_color,...
    'MarkerFaceAlpha',0.2)

scatter3(tcr_binary_phos_x,tcr_binary_phos_y,100*ones(size(tcr_binary_phos_x)),...
    ms,...
    'MarkerEdgeColor','none',...
    'MarkerFaceColor',ptcr_color)
hold off

caxis([0 70])
view(2)
grid off
box on
axis equal
axis tight
axis([0 size(Z1,1) 0 size(Z1,2)])
caxis([0 70])
xticks([0:100:size(Z1,1)])
yticks([0:100:size(Z1,2)])
xticklabels(10*xticks)
yticklabels(10*yticks)
%% radial sum
array_size = size(Z1);
s_tcr = ones(size(linind_tcr));
s_cd45 = ones(size(linind_cd45));


linind_ptcr = find(tcr_binary_phos);
s_ptcr = ones(size(linind_ptcr));

[lck_x,lck_y,s_lck] = find(Lck_array');
linind_lck = sub2ind(array_size,lck_x,lck_y);
%%%
[tcr_rings_counts,tcr_normalyzed_rings_counts] = ...
    ringCounts(array_size,linind_tcr,s_tcr);

[ptcr_rings_counts,ptcr_normalyzed_rings_counts] = ...
    ringCounts(array_size,linind_ptcr,s_ptcr);

[cd45_rings_counts,cd45_normalyzed_rings_counts] = ...
    ringCounts(array_size,linind_cd45,s_cd45);

[lck_rings_counts,lck_normalyzed_rings_counts] = ...
    ringCounts(array_size,linind_lck,s_lck);
%%
fig46 = figure(46);
set(fig46,'Position',fig_position.*[1,1,0.5,1.0])
Rs = 1:length(tcr_rings_counts);
face_alpha = 1.0;

bar(Rs,tcr_normalyzed_rings_counts,1,...
    'EdgeColor','none',...
    'FaceColor',tcr_color,'FaceAlpha',face_alpha)
hold on
bar(Rs,cd45_normalyzed_rings_counts,1,...
    'EdgeColor','none',...
    'FaceColor',cd45_color,'FaceAlpha',face_alpha)
hold off
axis([0 array_size(1)/2 0 0.3])


fig47 = figure(47);
set(fig47,'Position',fig_position.*[1.1,1.1,0.5,1.0])
Rs = 1:length(tcr_rings_counts);
face_alpha = 0.5;
bar(Rs,0.0025*lck_normalyzed_rings_counts,1,...
    'EdgeColor','none',...
    'FaceColor',lck_color,'FaceAlpha',0.5)
hold on
bar(Rs,tcr_normalyzed_rings_counts,1,...
    'EdgeColor','none',...
    'FaceColor',tcr_color,'FaceAlpha',face_alpha)
% bar(Rs,ptcr_normalyzed_rings_counts,1,...
%     'FaceColor',ptcr_color,'FaceAlpha',face_alpha)
bar(Rs,cd45_normalyzed_rings_counts,1,...
    'EdgeColor','none',...
    'FaceColor',cd45_color,'FaceAlpha',face_alpha)
hold off
axis([0 array_size(1)/2 0 0.3])


fig48 = figure(48);
set(fig48,'Position',fig_position.*[1.2,1.2,0.5,1.0])
Rs = 1:length(tcr_rings_counts);
face_alpha = 0.5;


bar(Rs,tcr_normalyzed_rings_counts,1,...
    'EdgeColor','none',...
    'FaceColor',tcr_color,'FaceAlpha',face_alpha)
hold on
bar(Rs,cd45_normalyzed_rings_counts,1,...
    'EdgeColor','none',...
    'FaceColor',cd45_color,'FaceAlpha',face_alpha)
bar(Rs,ptcr_normalyzed_rings_counts,1,...
    'EdgeColor','none',...
    'FaceColor',ptcr_color,'FaceAlpha',1.0)

hold off
axis([0 array_size(1)/2 0 0.3])

%% plot tcr, cd45

%}