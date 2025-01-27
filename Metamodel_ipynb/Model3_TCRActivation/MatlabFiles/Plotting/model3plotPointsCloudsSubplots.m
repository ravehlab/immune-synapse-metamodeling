function model3plotPointsCloudsSubplots(Nrows,Ncols,samples_data)

% molecules colors: %%%%%%%%%%%%%%%%%%%
TCR_color = [0.0, 0.6, 0.0];
CD45_color = [1.0, 0.0, 0.0];

% subplots and gaps sizes (relative to the figure size):
gapx0 = 0.12;
gapy0 = 0.125;
gapx = 0.03;
gapy = 0.02;
sx = 0.26;
sy = 0.14;

% origins of the individual subplots:
ox = gapx0 + [0, 1*(sx+gapx), 2*(sx+gapx)];
oy = gapy0 + [0, 1*(sy+gapy), 2*(sy+gapy),...
                 3*(sy+gapy), 4*(sy+gapy)];

%% array sizes: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
array_size_x_microns = 2; % 4
array_size_y_microns = 2; % 4

array_size_x_pixels = 200;
array_size_y_pixels = 200;

%% TCR_locations %%%%%%%%%%%%%%%%%%%%%%
TCR_cluster_density = 1000;
TCR_r1 = 0;
TCR_r2 = 0.25; % microns
pixel_size = 10; % nm

%%% create TCR and CD45 locations: %%%%%%%%%%%%%%
%% TCR: 
[TCR_x_pixels0,TCR_y_pixels0] = radialDistributionArray(...
    TCR_cluster_density,TCR_r1,TCR_r2,pixel_size,...
    array_size_x_microns,array_size_y_microns);
TCR_x_pixels = TCR_x_pixels0; % - 100*array_size_x_microns/2;
TCR_y_pixels = TCR_y_pixels0; % - 100*array_size_y_microns/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(14)
clf
set(gcf, 'Units', 'pixels',...
        'OuterPosition', [700, 50, 420, 800]);
    
axis_off = 0;
% clf
 for s2 = 1:Ncols*Nrows
    %%%%%%%%%%%%%%%%%%%%%%
    % s2 to row index: 
    col_ind = ceil(s2/Nrows);
    % s2 to column index: 
    row_ind = 1+mod(s2+4,Nrows); 

    % index of depletion value:
    idep = 0.1*samples_data(s2).dep;
    % index of decay_length value:
    idec = 0.1*samples_data(s2).dec;
    % name (letter) of point
    iname = samples_data(s2).name;
        
    pos1 = ([ox(col_ind), oy(6-row_ind), sx, sy]);
    h(s2) = subplot('Position',pos1);
    
    %% calculate locations: %%%%%%%%%%%%%%%%%%%%%
    %% CD45: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    depletion_range_nm = samples_data(s2).dep;

    CD45_cluster_density = 1000;
    CD45_r1 = TCR_r2 + depletion_range_nm/1000; % nm
    CD45_r2 = CD45_r1 + 0.3; % nm 0.3

    [CD45_x_pixels0,CD45_y_pixels0] = radialDistributionArray(...
        CD45_cluster_density,CD45_r1,CD45_r2,pixel_size,...
        array_size_x_microns,array_size_y_microns);
    CD45_x_pixels = CD45_x_pixels0; % - 100*array_size_x_microns/2;
    CD45_y_pixels = CD45_y_pixels0; % - 100*array_size_y_microns/2;
    
    %% phosphorylation probability: %%%%%%%%%%%%%
%     x_pixels = -100:1:100;
    R_max = 100;
    %%% CD45 decay:
    CD45_decay_length_nm = 10;
    CD45_decay_disk = decayDisk(...
        CD45_decay_length_nm,pixel_size,R_max);
    norm_CD45_decay_disk = CD45_decay_disk/sum(sum(CD45_decay_disk));

    %%% aLck decay:
    aLck_decay_length_nm = samples_data(s2).dec; %100;

    aLck_decay_disk = decayDisk(...
        aLck_decay_length_nm,pixel_size,R_max);
    norm_aLck_decay_disk = aLck_decay_disk/sum(sum(aLck_decay_disk));

    %%% sum of decays:
    sum_norm_decay_disk = norm_aLck_decay_disk - norm_CD45_decay_disk;
    sum_norm_decay_disk(sum_norm_decay_disk < 0) = 0;
    %%%%
    %%%%
    decay_probability_array = aLckProbabilityArray(...
    sum_norm_decay_disk,array_size_x_pixels,array_size_y_pixels,...
    CD45_x_pixels0,CD45_y_pixels0);
    
    %% plotting: %%%%%%%%%%%%%%%%%%%%%%
    %% points: %%%%%%%%%%%%%%%%%%%%%%%%
    plot(TCR_x_pixels,TCR_y_pixels,'.','Color',TCR_color,...
        'MarkerSize',1)
    hold on
    plot(CD45_x_pixels,CD45_y_pixels,'.','Color',CD45_color,...
        'MarkerSize',1)
    axis equal
    %% clouds: %%%%%%%%%%%%%%%%%%%%%%%%
    h2 = surf(decay_probability_array - 1);
    hold off

    h2.EdgeColor = 'none';
    h2.FaceAlpha = 1.0;
    orange_blue_colormap = orangeBlueColormap(64);
    colormap(orange_blue_colormap)
    view(2)
    
    if axis_off
        set(gca,'xtick',[],'ytick',[])
        set(gca,'xlabel',[],'ylabel',[])
    end

    
    

%     axis([0 200 0 200])
    axis([50 200-50 50 200-50])
    xticks([50:50:150])
    yticks([50:50:150])
    xticklabels({-1000 + 10*[50:50:150]})
    yticklabels({-1000 + 10*[50:50:150]})
%     axis tight
    axis equal
    %%
    if axis_off
        set(gca,'xtick',[],'ytick',[])
        set(gca,'xlabel',[],'ylabel',[])
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    text(10,180,iname,'FontWeight', 'Bold','Color','w')
%     xticks([]);
%     yticks([]);
 end

% delete XTickLabels:
set([h(1),h(2),h(3),h(4)],'XTickLabel','');
set([h(6),h(7),h(8),h(9)],'XTickLabel','');
set([h(11),h(12),h(13),h(14)],'XTickLabel','');
% delete YTickLabels:
set([h(6),h(7),h(8),h(9),h(10),...
     h(11),h(12),h(13),h(14),h(15)],'YTickLabel','');

% yticks([0 0.1 0.2])
% if ismember(s2,[13,14,15])
%     xlabel('x(nm)')
% end
% if ismember(s2,[1,4,7,10,13])
%     ylabel('y(nm)')
% end


end