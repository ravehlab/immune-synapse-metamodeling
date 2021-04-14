function plotIntro()

%% doc: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{

Input:
Calls:
Output:

%}

%% molecules colors: %%%%%%%%%%%%%%%%%%
TCR_color  = [0.0, 0.6, 0.0];
CD45_color = [1.0, 0.0, 0.0];
pTCR_color = [1.0, 0.0, 1.0];
aLck_color = [1.0, 0.5, 0.0];
points_alpha = 0.3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nc = 64;
magenta_fixed_colormap = magentaFixedColormap(Nc);
orange_fixed_colormap = orangeFixedColormap(Nc);
colormap(orange_fixed_colormap)
%
%% array sizes: %%%%%%%%%%%%%%%%%%%%%%%
array_size_x_microns = 2;
array_size_y_microns = 2;

pixel_size_nm = 10; % nm

array_size_x_pixels = array_size_x_microns*100;
array_size_y_pixels = array_size_y_microns*100;

s_pixels = array_size_x_pixels;
lim1 = 60; % nm
lim2 = 30; % nm
tick1 = 50;
tick2 = 25;
%
%% subplots locations: %%%%%%%%%%%%%%%%
axis_off = 1;

gapx0 = 0.08;
gapy0 = 0.125;
gapx = 0.04;
gapy = 0.01;
sx = 0.25;
sy1 = 0.5;
sy2 = 0.25;

% origins of the individual subplots:
ox = gapx0 + [0, 1*(sx+gapx), 2*(sx+gapx)];
oy = gapy0 + [0, 1*(sy2+gapy)];

pos1 = ([ox(1), oy(2), sx, sy1]);
pos2 = ([ox(2), oy(2), sx, sy1]);
pos3 = ([ox(3), oy(2), sx, sy1]);
pos4 = ([ox(1), oy(1), sx, sy2]);
pos5 = ([ox(2), oy(1), sx, sy2]);
pos6 = ([ox(3), oy(1), sx, sy2]);

add_TCR = 1;
add_CD45 = 1;
ms1 = 5;
ms2 = 5;
%
%% start subplots loops: %%%%%%%%%%%%%%
depletions = [-250,0:10:200];
decayLengths = 10:10:200;

% selected indices to plot as subplots:
s_dep_ind = [1,2:5:22]; % selected, N = N_rows
s_dec_ind = [2,5,10,15,20]; % selected, N = N_cols

deps = depletions(s_dep_ind([1,4,3]));
decs = decayLengths(s_dec_ind([5,5,2]));

% deps = [-250,100,50]; % nm
% decs = [200,200,50]; % nm
%% generate TCR locations: %%%%%%%%%%%%
TCR_cluster_density = 1000;
TCR_r1 = 0;
TCR_r2 = 0.25; % microns

% TCR locations: %%%%%%%%%%%%%%%%%%%%%
[TCR_x_pixels0,TCR_y_pixels0] = radialDistributionArray(...
    TCR_cluster_density,TCR_r1,TCR_r2,pixel_size_nm,...
    array_size_x_microns,array_size_y_microns);
TCR_x_pixels = TCR_x_pixels0;
TCR_y_pixels = TCR_y_pixels0;

%%% TCR_locations_array:
TCR_locations_array = zeros(array_size_x_pixels,...
    array_size_x_pixels);
linind_TCR = sub2ind(size(TCR_locations_array),...
    TCR_x_pixels,TCR_y_pixels);
TCR_locations_array(linind_TCR) = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% decay_disk of one CD45: %%%%%%%%%%%%
% lambda = 0.01; % 1/nm
x_pixels = -ceil(array_size_x_pixels/2):1:ceil(array_size_x_pixels/2);
R_max = 100;

%%% CD45 decay:
CD45_decay_length_nm = 10;

CD45_double_decay = model3DoubleDecay(...
    CD45_decay_length_nm, pixel_size_nm, x_pixels);
norm_CD45_double_decay = CD45_double_decay/sum(CD45_double_decay);

CD45_decay_disk = decayDisk(...
    CD45_decay_length_nm,pixel_size_nm,R_max);
norm_CD45_decay_disk = CD45_decay_disk/sum(sum(CD45_decay_disk));


for ind = 1:3
    
    depletion_range_nm = deps(ind);
    aLck_decay_length_nm = decs(ind);

    CD45_cluster_density = 1000;
    CD45_r1 = TCR_r2 + depletion_range_nm/1000; % nm
    CD45_r2 = CD45_r1 + 0.3; % nm 0.3
    %
    %% CD45 locations: %%%%%%%%%%%%%%%%
    [CD45_x_pixels0,CD45_y_pixels0] = radialDistributionArray(...
        CD45_cluster_density,CD45_r1,CD45_r2,pixel_size_nm,...
        array_size_x_microns,array_size_y_microns);
    CD45_x_pixels = CD45_x_pixels0;% - 100*array_size_x_microns/1;
    CD45_y_pixels = CD45_y_pixels0;% - 100*array_size_y_microns/1;
    %
    %% CD45_locations_array: %%%%%%%%%%
    CD45_locations_array = zeros(array_size_x_pixels,...
        array_size_x_pixels);
    linind_CD45 = sub2ind(size(CD45_locations_array),...
        CD45_x_pixels,CD45_y_pixels);
    CD45_locations_array(linind_CD45) = 1;
    %
    %% cross section TCR and CD45: %%%%
    %%% TCR:
    TCR_normalized_counts  = sumOverRings(TCR_locations_array);
    TCR_normalized_counts(3) = 0.5*(TCR_normalized_counts(2)+...
        TCR_normalized_counts(4));
    TCR_normalized_counts(1) = 0.5*(TCR_normalized_counts(2)+...
        TCR_normalized_counts(3));

    double_TCR_normalized_counts = ...
        [fliplr(TCR_normalized_counts'),...
        TCR_normalized_counts(1),...
        TCR_normalized_counts'];

    %%% CD45:
    CD45_normalized_counts  = sumOverRings(CD45_locations_array);
    double_CD45_normalized_counts = ...
        [fliplr(CD45_normalized_counts'),...
        CD45_normalized_counts(1),...
        CD45_normalized_counts'];

    %
    %% aLck decay: %%%%%%%%%%%%%%%%%%%%
    aLck_decay_disk = decayDisk(...
        aLck_decay_length_nm,pixel_size_nm,R_max);
    norm_aLck_decay_disk = aLck_decay_disk/sum(sum(aLck_decay_disk));

    %%% sum of decays:
    sum_norm_decay_disk = norm_aLck_decay_disk - norm_CD45_decay_disk;
    sum_norm_decay_disk(sum_norm_decay_disk < 0) = 0;

    %%% radial sum one disk: %%%%%%%%%%%%%%%%%%%%%%%%%
    sum_norm_decay_disk_normalized_counts = sumOverRings(sum_norm_decay_disk);
    double_aLck_decay_disk_normalized_counts = ...
        [fliplr(sum_norm_decay_disk_normalized_counts'),...
        sum_norm_decay_disk_normalized_counts(1),...
        sum_norm_decay_disk_normalized_counts'];
    %
    %% aLck_probability_array %%%%%%%%%
    decay_probability_array = aLckProbabilityArray(...
        sum_norm_decay_disk,array_size_x_pixels,array_size_y_pixels,...
        CD45_x_pixels,CD45_y_pixels);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% sum over rings of decay array: %
    decay_normalized_counts = sumOverRings(decay_probability_array);
    double_decay_normalized_counts = ...
        [fliplr(decay_normalized_counts'),...
        decay_normalized_counts(1),...
        decay_normalized_counts'];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% locations to array: %%%%%%%%%%%%
    %%% pTCR_probability_array:
    pTCR_probability_array = decay_probability_array.*...
        TCR_locations_array;

    %%% TCR phosphorylation hist %%%%%%%%%%%
    % TCR_normalized_counts  = sumOverRings(TCR_locations_array);
    pTCR_normalized_counts = sumOverRings(pTCR_probability_array);
    rTCR_normalized_counts = pTCR_normalized_counts./...
        TCR_normalized_counts;
    rTCR_normalized_counts(isnan(rTCR_normalized_counts)) = 0;
    % rTCR_normalized_counts(1)
    %
    %% plot and subplots sizes: %%%%%%%
    figure(ind)
    clf
    set(gcf, 'Units', 'pixels',...
        'OuterPosition', [200 + 30*ind, 50+30*ind, 800, 600]);
    
    ax = axes;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% subplot(2,3,1) TCR and CD45 location: 
    ax(1) = subplot('Position',pos1);
    plot(TCR_x_pixels,TCR_y_pixels,'.','Color',TCR_color,...
        'MarkerSize',ms1)
    hold on
    plot(CD45_x_pixels,CD45_y_pixels,'.','Color',CD45_color,...
        'MarkerSize',ms1)
    hold off
%     legend('TCR','CD45')
    axis equal
    axis(s_pixels/2 + [-lim1 lim1 -lim1 lim1])
    xticks(s_pixels/2 + [-tick1:tick1:tick1])
    yticks(s_pixels/2 + [-tick1:tick1:tick1])
    xticklabels({10*[-tick1:tick1:tick1]})
    yticklabels({10*[-tick1:tick1:tick1]})
%     xlabel('x(nm)')
%     ylabel('y(nm)')
    if axis_off
        set(gca,'xtick',[],'ytick',[])
        set(gca,'xlabel',[],'ylabel',[])
    end
    %
    %% subplot(2,3,4): %%%%%%%%%%%%%%%%
    h(4) = subplot('Position',pos4);

    bar(-ceil(array_size_x_pixels/2):1:ceil(array_size_x_pixels/2),...
        double_TCR_normalized_counts,0.9,...
        'FaceColor',TCR_color,'EdgeColor','none')
    hold on
    bar(-ceil(array_size_x_pixels/2):1:ceil(array_size_x_pixels/2),...
        double_CD45_normalized_counts,0.9,...
        'FaceColor',CD45_color,'EdgeColor','none')
    hold off

    axis([-lim1 lim1 0 0.2])
    xticks([-tick1:tick1:tick1])
    xticklabels({10*[-tick1:tick1:tick1]})
    xlabel('x(nm)')

    if axis_off
        set(gca,'xtick',[],'ytick',[])
        set(gca,'xlabel',[],'ylabel',[])
    end
    %
    %% subplot(2,3,2), surf aLck_probability_array:
    %     h(2) = subplot('Position',pos2);
    ax(2) = subplot('Position',pos2);
    sp2 = surf(decay_probability_array-1);
    colormap(ax(2),orange_fixed_colormap);
%     colormap(magenta_fixed_colormap)
    sp2.EdgeColor = 'none';
    sp2.FaceAlpha = 1.0;
    alpha color
    alpha scaled
    
    if add_TCR
        hold on
        scatter(TCR_x_pixels,TCR_y_pixels,ms2,...
            'MarkerEdgeColor','none',...
            'MarkerFaceColor',TCR_color,...
            'MarkerFaceAlpha',points_alpha);
        hold off
    end
    if add_CD45
        hold on
        scatter(CD45_x_pixels,CD45_y_pixels,ms2,...
            'MarkerEdgeColor','none',...
            'MarkerFaceColor',CD45_color,...
            'MarkerFaceAlpha',points_alpha);
        hold off
    end
    
    axis equal
    axis(s_pixels/2 + [-lim1 lim1 -lim1 lim1])
    xticks(s_pixels/2 + [-tick1:tick1:tick1])
    yticks(s_pixels/2 + [-tick1:tick1:tick1])
    xticklabels({pixel_size_nm*[-tick1:tick1:tick1]})
    yticklabels({10*[-tick1:tick1:tick1]})
%     xlabel('x(nm)')
%     ylabel('y(nm)')

    box on
    view(2)
    if axis_off
        set(gca,'xtick',[],'ytick',[])
        set(gca,'xlabel',[],'ylabel',[])
    end
    %
    %% subplot(2,3,5): %%%%%%%%%%%%%%%%%%%%
    h(5) = subplot('Position',pos5);

    if add_TCR
        hold on
        bar(-s_pixels/2:1:s_pixels/2,double_TCR_normalized_counts,0.9,...
            'FaceColor',TCR_color,'EdgeColor','none',...
            'FaceAlpha',points_alpha)
        hold off

    end
    if add_CD45
        hold on
        bar(-s_pixels/2:1:s_pixels/2,double_CD45_normalized_counts,0.9,...
            'FaceColor',CD45_color,'EdgeColor','none',...
            'FaceAlpha',points_alpha)
        hold off
    end
    hold on
    bar(-ceil(s_pixels/2):1:ceil(s_pixels/2),...
        double_decay_normalized_counts,0.9,...
        'FaceColor',aLck_color,'EdgeColor','none')
    hold off
    axis([-lim1 lim1 0 0.2])
    xticks([-tick1:tick1:tick1])
    xticklabels({10*[-tick1:tick1:tick1]})
    xlabel('x(nm)')
    box on

    if axis_off
        set(gca,'xtick',[],'ytick',[])
        set(gca,'xlabel',[],'ylabel',[])
    end
    %
    %% subplot(2,3,3): %%%%%%%%%%%%%%%%%%%%
%     sp3 = subplot('Position',pos3);
    ax(3) = subplot('Position',pos3);
%     sp3 = pcolor(pTCR_probability_array');
    sp3 = pcolor(pTCR_probability_array');
    colormap(ax(3),magenta_fixed_colormap);
%     colormap(magenta_fixed_colormap)
    sp3.EdgeColor = 'none';
    alpha color
    alpha scaled
    axis equal
    axis tight
    axis(s_pixels/2 + [-lim2 lim2 -lim2 lim2])
    xticks([-tick2:tick2:tick2])
    yticks([-tick2:tick2:tick2])
    xticklabels({pixel_size_nm*[-tick2:tick2:tick2]})
    yticklabels({pixel_size_nm*[-tick2:tick2:tick2]})
    xlabel('x(nm)')
    ylabel('y(nm)')

    if axis_off
        set(gca,'xtick',[],'ytick',[])
        set(gca,'xlabel',[],'ylabel',[])
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% subplot(2,3,6): %%%%%%%%%%%%%%%%%%%%
    h(6) = subplot('Position',pos6);
%     rTCR_normalized_counts(4) = rTCR_normalized_counts(5);
    rTCR_normalized_counts(3) = rTCR_normalized_counts(4);
    rTCR_normalized_counts(2) = rTCR_normalized_counts(3);
    rTCR_normalized_counts(1) = rTCR_normalized_counts(3);
%     
    double_rTCR_normalized_counts = ...
        [fliplr(rTCR_normalized_counts'),...
        rTCR_normalized_counts(1),...
        rTCR_normalized_counts'];
    bar(-pixel_size_nm:1:pixel_size_nm,...
        1*double_rTCR_normalized_counts,0.9,'FaceColor', pTCR_color)

    axis([-lim2 lim2 0 0.12])
    % axis square
    xticks([-tick2:tick2:tick2])
    xticklabels({pixel_size_nm*[-tick2:tick2:tick2]})
    xlabel('r(nm)')

    if axis_off
        set(gca,'xtick',[],'ytick',[])
        set(gca,'xlabel',[],'ylabel',[])
    end
    %
end
%{
ax1 = axes;
[x,y,z] = peaks;
surf(ax1,x,y,z)
view(2)
ax2 = axes;
scatter(ax2,randn(1,120),randn(1,120),50,randn(1,120),'filled')

%%Link them together
linkaxes([ax1,ax2])
%%Hide the top axes
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];
%%Give each one its own colormap
colormap(ax1,'hot')
colormap(ax2,'cool')
%%Then add colorbars and get everything lined up
set([ax1,ax2],'Position',[.17 .11 .685 .815]);
cb1 = colorbar(ax1,'Position',[.05 .11 .0675 .815]);
cb2 = colorbar(ax2,'Position',[.88 .11 .0675 .815]);
%}


