function model3plotTCRphosHistArray(rTCR_hist_model3,...
    rTCR_hist_model3_negativeDep)

%% doc: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{


Input:
Calls: orangeGrayColormap,
       parulaGrayColormap,
       radialDistributionArray,
       
Output:

%}
%

%% rad rTCR hist data: %%%%%%%%%%%%%%%%
r = 1:size(rTCR_hist_model3,3);

%% colors and colormaps: %%%%%%%%%%%%%%
aLck_color = [1.0, 0.5, 0];
pTCR_color = [1.0, 0.0, 1.0];

Nc = 64;
orange_fixed_colormap = orangeFixedColormap(Nc);
magenta_fixed_colormap = magentaFixedColormap(Nc);
%
%% sizes: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_rows = 6;
N_cols = 5;

% subplots and gaps sizes (relative to the figure size):
gapx0 = 0.12; % relative initial gap on the left.
gapy0 = 0.125; % relative initial gap on the bottom.
gapx = 0.02; % relative gap x between subplots.
gapy = 0.02; % relative gap x between subplots.
size_x = 0.75/N_cols; % relative subplots x size.
size_y = 0.75/N_rows; % relative subplots y size.

% origins of the individual subplots:
origins_x = gapx0 + (size_x+gapx)*[0:N_cols-1];
origins_y = gapy0 + (size_y+gapy)*[0:N_rows-1];

%
%% array sizes: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
array_size_x_microns = 2;
array_size_y_microns = 2;

array_size_x_pixels = array_size_x_microns*100;
array_size_y_pixels = array_size_y_microns*100;
%
% %% TCR_locations %%%%%%%%%%%%%%%%%%%%%%
% TCR_cluster_density = 1000;
% TCR_r1 = 0;
% TCR_r2 = 0.25; % microns
% pixel_size = 10; % nm
% 
% %%% create TCR locations: %%%%%%%%%%%%%
% [TCR_x_pixels0,TCR_y_pixels0] = radialDistributionArray(...
%     TCR_cluster_density,TCR_r1,TCR_r2,pixel_size,...
%     array_size_x_microns,array_size_y_microns);
% 
% TCR_x_pixels = TCR_x_pixels0 - array_size_x_pixels/2;
% TCR_y_pixels = TCR_y_pixels0 - array_size_y_pixels/2;

%% start subplots loops: %%%%%%%%%%%%%%
depletions = [-250,0:10:200];
decayLengths = 10:10:200;

% selected indices to plot as subplots:
s_dep_ind = [1:5:21]; % selected, N = N_rows
s_dec_ind = [2,5,10,15,20]; % selected, N = N_cols

%% plot clouds: %%%%%%%%%%%%%%%%%%%%%%%
figure(15)
clf
set(gcf, 'Units', 'pixels',...
        'OuterPosition', [600, 50, 700, 800]);
    
axis_off = 1;
lim1 = 30; % 100
tick1 = 25; 
max_h = 0.12;

for col_ind = 1:N_cols
    for row_ind = 1:N_rows-1

        % index of depletion value:
        idep = s_dep_ind(row_ind);
        % index of decay_length value:

        idec = s_dec_ind(col_ind);

        % name (letter) of point
        % iname = samples_data(s2).name;

        pos1 = ([origins_x(col_ind),...
                 origins_y(row_ind+1),...
                 size_x, size_y]); % N_rows+1 - 

        h1 = subplot('Position',pos1);

        %% plot TCR phos hist: %%%%%%%%
        % 3D array to 1D:
        rTCR_hist = squeeze(rTCR_hist_model3(idep,idec,:));
        rr2 = [-fliplr(r),0,r];
        rTCR_hist2 = [flipud(rTCR_hist);0;rTCR_hist];
        rTCR_hist2(length(r)-2:1:length(r)+2) = ...
            rTCR_hist(2);

        % bar plot:
        bar(rr2, rTCR_hist2,1,'FaceColor',pTCR_color,...
            'EdgeColor','none')    
        %% mark Exp and Mean %%%%%%%%%%%%%%%%%%%%
        % rTCR_Exp:
        rTCR_Exp = sum(rTCR_hist.*r')/sum(rTCR_hist);

        % rTCR_mean:
        % TODO: fix this (25-1)
        rTCR_mean = sum(rTCR_hist)/(25-1); % !!!

        hold on
        % plot location of 

        plot([-25 25],...
            [rTCR_mean rTCR_mean],'-',...
            'Color',[0.0, 0.0, 1.0],'LineWidth',0.5)

        plot(rTCR_Exp,0.01,'v',...
            'MarkerSize',6,...
            'MarkerEdgeColor',[0.0, 0.0, 1.0],...
            'MarkerFaceColor',[0.0, 0.0, 1.0])

        hold off
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        xlim([-lim1 lim1])
        ylim([0 max_h])
        xticks([-tick1:tick1:tick1]);
        xticklabels(10*[-tick1:tick1:tick1])

        %%
        if axis_off
            set(gca,'xtick',[],'ytick',[])
            set(gca,'xlabel',[],'ylabel',[])
        end

    end
end

%% negative dep: %%%%%%%%%%%%%%%%%%%%%%

for col_ind = 1:N_cols
    for row_ind = 1:1 %N_rows-1:N_rows-1

        % index of depletion value:
%         idep = s_dep_ind(row_ind);
        % index of decay_length value:

        idec = s_dec_ind(col_ind);

        % name (letter) of point
        % iname = samples_data(s2).name;

        pos1 = ([origins_x(col_ind),...
                 origins_y(row_ind+0),...
                 size_x, size_y]); % N_rows+1 - 

        h1 = subplot('Position',pos1);

        %% plot TCR phos hist: %%%%%%%%
        % 3D array to 1D:

        rTCR_hist = squeeze(rTCR_hist_model3_negativeDep(1,idec,:));
        rr2 = [-fliplr(r),0,r];
        rTCR_hist2 = [flipud(rTCR_hist);0;rTCR_hist];
        rTCR_hist2(length(r)-2:1:length(r)+2) = ...
            rTCR_hist(2);
        
        % bar plot:
        bar(rr2, rTCR_hist2,1,'FaceColor',pTCR_color,...
            'EdgeColor','none')    
        %% mark Exp and Mean %%%%%%%%%%%%%%%%%%%%
        % rTCR_Exp:
        rTCR_Exp = sum(rTCR_hist.*r')/sum(rTCR_hist);

        % rTCR_mean:
        % TODO: fix this (25-1)
        rTCR_mean = sum(rTCR_hist)/(25-1); % !!!

        hold on
        % plot location of mean h:
        plot([-25 25],...
            [rTCR_mean rTCR_mean],'-',...
            'Color',[0.0, 0.0, 1.0],'LineWidth',0.5)
        
        % plot location of mean r:
        plot(rTCR_Exp,0.01,'v',...
            'MarkerSize',6,...
            'MarkerEdgeColor',[0.0, 0.0, 1.0],...
            'MarkerFaceColor',[0.0, 0.0, 1.0])

        hold off
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         text(-27,0.05,iname,'FontWeight', 'Bold')
        xlim([-lim1 lim1])
        ylim([0 max_h]) % 0.06
        xticks([-25:25:25]);
        xticklabels(10*[-25:25:25])

        %%
        if axis_off
            set(gca,'xtick',[],'ytick',[])
            set(gca,'xlabel',[],'ylabel',[])
        end

    end
end
%
end
