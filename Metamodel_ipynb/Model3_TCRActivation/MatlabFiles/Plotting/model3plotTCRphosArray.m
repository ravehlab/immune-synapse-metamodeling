function model3plotTCRphosArray(parameters)

%% doc: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{


Input:
Calls: orangeGrayColormap,
       parulaGrayColormap,
       radialDistributionArray,
       
Output:

%}
%
%% colors and colormaps: %%%%%%%%%%%%%%
TCR_color = parameters.TCR.color;
CD45_color = parameters.CD45.color;

orange_fixed_colormap = parameters.colormaps.orange_fixed;
%
%% sizes: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subplots and gaps sizes (relative to the figure size):
N_cols = parameters.plots.N_cols;
N_rows = parameters.plots.N_rows;

gapx0 = parameters.plots.gapx0; % relative initial gap on the left.
gapy0 = parameters.plots.gapy0; % relative initial gap on the bottom.
gapx = parameters.plots.gapx; % relative gap x between subplots.
gapy = parameters.plots.gapy; % relative gap x between subplots.
size_x = parameters.plots.size_x; % relative subplots x size.
size_y = parameters.plots.size_y; % relative subplots y size.

% origins of the individual subplots:
origins_x = gapx0 + (size_x + gapx)*[0:N_cols-1];
origins_y = gapy0 + (size_y + gapy)*[0:N_rows-1];
%
%% array sizes: %%%%%%%%%%%%%%%%%%%%%%%
array_size_x_microns = parameters.arrays.size_x_microns;
array_size_y_microns = parameters.arrays.size_y_microns;

pixel_size_nm = parameters.arrays.pixel_size_nm; % nm
mic2nm = parameters.arrays.mic2nm;

array_size_x_pixels = array_size_x_microns*mic2nm/pixel_size_nm;
array_size_y_pixels = array_size_y_microns*mic2nm/pixel_size_nm;

shift_x_pixels = ceil(array_size_x_pixels/2);
shift_y_pixels = ceil(array_size_y_pixels/2);
%
%% molecules parameters: %%%%%%%%%%%%%%
% TCR:
TCR_cluster_density = parameters.TCR.cluster_density;
TCR_r1_microns = parameters.TCR.r1_microns;
TCR_r2_microns = parameters.TCR.r2_microns; % microns

% CD45:
CD45_cluster_density = parameters.CD45.cluster_density; % #/micron^2
CD45_decay_length_nm = parameters.CD45.decay_length_nm; 
CD45_width_microns1 = parameters.CD45.width_microns; % ring width
CD45_width_microns2 = parameters.CD45.width_microns2; % ring width

%%% create TCR locations: %%%%%%%%%%%%%
[TCR_x_pixels0,TCR_y_pixels0] = radialDistributionArray(...
    TCR_cluster_density,TCR_r1_microns,TCR_r2_microns,pixel_size_nm,...
    array_size_x_microns,array_size_y_microns);

TCR_x_pixels = TCR_x_pixels0 - array_size_x_pixels/2;
TCR_y_pixels = TCR_y_pixels0 - array_size_y_pixels/2;
%
%% plot TCR and CD45 locations: %%%%%%%
figure(14)
clf
set(gcf, 'Units', 'pixels',...
        'OuterPosition', [400, 50, 700, 800]);
    
lim1 = parameters.plots.lim1; % pixels
tick1 = parameters.plots.tick1; % pixels
lim2 = parameters.plots.lim2; % pixels
tick2 = parameters.plots.tick2; % pixels

depletions = parameters.plots.depletions;
decayLengths = parameters.plots.decayLengths;

% selected indices to plot as subplots:
s_dep_ind = parameters.plots.s_dep_ind; % selected, N = N_rows
s_dec_ind = parameters.plots.s_dec_ind; % selected, N = N_cols

axis_off = parameters.plots.axis_off;
R_max = ceil(array_size_x_pixels/1);
%
%% start subplots loops: %%%%%%%%%%%%%%


for col_ind = 1:N_cols
    for row_ind = 1:N_rows

        % index of depletion value:
        idep = s_dep_ind(row_ind);
        % index of decay_length value:

        idec = s_dec_ind(col_ind);

        % name (letter) of point
    %     iname = samples_data(s2).name;

        pos1 = ([origins_x(col_ind),...
                 origins_y(row_ind),...
                 size_x, size_y]); % N_rows+1 - 

        h1 = subplot('Position',pos1);

        %% calculate locations: %%%%%%%%%%%%%%%%%%%%%
        %% CD45: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        depletion_range_nm = depletions(idep);

        if row_ind == 1
            CD45_width_microns = CD45_width_microns2;
        else
            CD45_width_microns = CD45_width_microns1;
        end
        CD45_r1_microns = TCR_r2_microns + depletion_range_nm/mic2nm; % nm
        CD45_r2_microns = CD45_r1_microns + CD45_width_microns; % nm 0.3

        [CD45_x_pixels0,CD45_y_pixels0] = radialDistributionArray(...
            CD45_cluster_density,CD45_r1,CD45_r2,pixel_size,...
            array_size_x_microns,array_size_y_microns);

        CD45_x_pixels = CD45_x_pixels0 - array_size_x_pixels/2;
        CD45_y_pixels = CD45_y_pixels0 - array_size_y_pixels/2;

        %% calculate clouds (phosphorylation probability)
        % CD45 decay:
        CD45_decay_disk = decayDisk(...
            CD45_decay_length_nm,pixel_size,R_max);
        norm_CD45_decay_disk = CD45_decay_disk/sum(sum(CD45_decay_disk));

        %%% aLck decay:
        aLck_decay_length_nm = decayLengths(idec); 

        aLck_decay_disk = decayDisk(...
            aLck_decay_length_nm,pixel_size,R_max);
        norm_aLck_decay_disk = aLck_decay_disk/sum(sum(aLck_decay_disk));

        % sum of decays:
        sum_norm_decay_disk = norm_aLck_decay_disk - norm_CD45_decay_disk;

        % phosphorylayion probability > 0:
        sum_norm_decay_disk(sum_norm_decay_disk < 0) = 0;

        % 2D covolution of decay_disk with CD45 locations:
        decay_probability_array = aLckProbabilityArray(...
        sum_norm_decay_disk,array_size_x_pixels,array_size_y_pixels,...
        CD45_x_pixels0,CD45_y_pixels0);

        %% calculate TCR phosphorylation:
        % TCR array: %%%%%%%%%%%%%%%%%%%%%
        TCR_array = zeros(array_size_x_pixels,...
            array_size_y_pixels);

        linind_TCR = sub2ind(size(TCR_array),...
            TCR_x_pixels0,TCR_y_pixels0);

        TCR_array(linind_TCR) = 1;
        %
        %% TCR phos probability: %%%%%%%%%%
        TCR_phos_probability_array  = ...
            TCR_array.*decay_probability_array;


        %% plot TCR phos: %%%%%%%%%%%%%
        h2 = pcolor(TCR_phos_probability_array);
        view(2)
        h2.EdgeColor = 'none';
        h2.FaceAlpha = 1.0;
%         colormap(magenta_fixed_colormap)
        colormap(orange_fixed_colormap)
        alpha color
        alpha scaled
        axis equal
        axis(shift_x + [-lim1 lim1 -lim1 lim1])
        xticks(shift_x + [-tick1:tick1:tick1])
        yticks(shift_y + [-tick1:tick1:tick1])
        xticklabels({pixel_size_nm*[-tick1:tick1:tick1]})
        yticklabels({pixel_size_nm*[-tick1:tick1:tick1]})
    %     xlabel('x(nm)')
    %     ylabel('y(nm)')
        %%
        if axis_off
            set(gca,'xtick',[],'ytick',[])
            set(gca,'xlabel',[],'ylabel',[])
        end

    end
end

end


