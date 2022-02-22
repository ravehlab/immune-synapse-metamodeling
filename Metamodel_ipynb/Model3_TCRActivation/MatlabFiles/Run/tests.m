% tests
% sq_dep_decay_rTCR_normalized_counts = ...
%     squeeze(dep_decay_rTCR_normalized_counts);
% 
% % rTCR_data = dlmread('rTCR_hist_model3_negativeDep.m');
% figure(6)
% % sq_dep_decay_TCR_normalized_counts = ...
% %     squeeze(dep_decay_TCR_normalized_counts);
% sq_dep_decay_rTCR_normalized_counts(...
%     isnan(sq_dep_decay_rTCR_normalized_counts)) = 0;
% 
% surf(sq_dep_decay_rTCR_normalized_counts)
% colorbar
% view(2)

%%
clear
clc

%%% setup for model3:
% molecules colors:
TCR_color  = [0.0, 0.6, 0.0];
CD45_color = [1.0, 0.0, 0.0];
aLCK_color = [1.0, 0.0, 1.0];
pTCR_color = [1.0, 0.5, 0.0];

%%% array size
array_size_x_microns = 2; %4;
array_size_y_microns = 2; %4;
pixel_size = 10; % nm
array_size_x_pixels = array_size_x_microns*1000/pixel_size;
array_size_y_pixels = array_size_y_microns*1000/pixel_size;

TCR_cluster_density = 1000; %N/um^2
TCR_r1 = 0; % microns
TCR_r2 = 0.25; % microns

CD45_cluster_density = 1000; %N/um^2
CD45_r1 = 0.45; % 0.45 microns
CD45_r2 = CD45_r1 + 0.3; % microns

%% % run_sample:
model3_run_sample;

%% % plot run_sample results:
model3_plot_run_sample;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% batch parameters:
batch_depletion_range_nm = -250; % 0:10:200; %0:20:300; % 0:50:750; % nm 100
batch_aLCK_decay_length_nm = 50; %0:10:200; %0:10:150; % 25:25:300; % nm 200

%%%
N_samples = 10;
N_bins = 100; %200;

%%% run over batch parameters: 
% model3_run;

N_dep   = length(batch_depletion_range_nm);
N_dec = length(batch_aLCK_decay_length_nm);

TCR_normalized_counts_array  =  zeros(N_samples,N_bins);
pTCR_normalized_counts_array =  zeros(N_samples,N_bins);

dep_decay_TCR_normalized_counts  = zeros(N_dep,N_dec,N_bins);
dep_decay_pTCR_normalized_counts = zeros(N_dep,N_dec,N_bins);

% N_bins = 200;
for dep_ind = 1:length(batch_depletion_range_nm)
    for dec_ind = 1:length(batch_aLCK_decay_length_nm)
        disp(dep_ind)
        disp(dec_ind)
        depletion_range_nm = batch_depletion_range_nm(dep_ind);
        decay_length_nm = batch_aLCK_decay_length_nm(dec_ind);
        
        %% decay_disk: %%%%%%%%%%%%%%%%%%%%%%%%
        decay_disk = model3_decayDisk(decay_length_nm,pixel_size,R_max);
        norm_decay_disk = decay_disk/sum(sum(decay_disk));
        
        CD45_decay_disk = model3_decayDisk(10,pixel_size,R_max);
        CD45_norm_decay_disk = CD45_decay_disk/sum(sum(CD45_decay_disk));
        
        sum_decay_disk = norm_decay_disk - CD45_norm_decay_disk;
        sum_decay_disk(sum_decay_disk < 0) = 0;
        
        %% CD45 ring:
        CD45_r1 = TCR_r2 + depletion_range_nm/1000; % nm
        CD45_r2 = CD45_r1 + 0.3; % nm 0.5
        
        for n = 1:N_samples
            disp(n)
            %% ramdom sampling
            %% TCR_locations %%%%%%%%%%%%%%%%%%%%%%
            [TCR_x_pixels,TCR_y_pixels] = radialDistributionArray(...
                TCR_cluster_density,TCR_r1,TCR_r2,pixel_size,...
                array_size_x_microns,array_size_y_microns);

            %% TCR_locations array %%%%%%%%%%%%%%%%
            TCR_locations_array = ...
                zeros(array_size_x_pixels,array_size_y_pixels);

            linind_TCR = sub2ind(size(TCR_locations_array),...
                TCR_x_pixels,TCR_y_pixels);

            TCR_locations_array(linind_TCR) = 1;

            %% CD45 locations %%%%%%%%%%%%%%%%%%%%%
            [CD45_x_pixels,CD45_y_pixels] = radialDistributionArray(...
                CD45_cluster_density,CD45_r1,CD45_r2,pixel_size,...
                array_size_x_microns,array_size_y_microns);
            
            %% plot points: %%%%%%%%%%%%%%%%%%%%%%%
            figure(1)
            plot(TCR_x_pixels,TCR_y_pixels,'.',...
            'Color',TCR_color)
            hold on
            plot(CD45_x_pixels,CD45_y_pixels,'.',...
            'Color',CD45_color)
            
%             hold off
            axis equal
            axis([50 150 50 150])
            % aLck_probability_array %%%%%%%%%%%%%
            aLck_probability_array = aLckProbabilityArray(...
                sum_decay_disk,array_size_x_pixels,array_size_y_pixels,...
                CD45_x_pixels,CD45_y_pixels); %sum_decay_disk,norm_decay_disk
            
            % CD45_probability_array %%%%%%%%%%%%%
            
            
            h1 = surf(aLck_probability_array' - 1);
            h1.EdgeColor = 'none';
            h1.FaceAlpha = 1.0;
            orange_blue_colormap = orangeBlueColormap(64);
            colormap(orange_blue_colormap)
            view(2)
            hold off
            %% TCR phosphorylation pattern %%%%%%%%
            pTCR_probability_array = TCR_locations_array.*aLck_probability_array;
            
            figure(2)
            
            h1 = surf(pTCR_probability_array' - 1);
            h1.EdgeColor = 'none';
            h1.FaceAlpha = 1.0;
            colormap default
            axis equal
            axis([50 150 50 150])
            
            view(2)
            
            
            %% sum over angles:
            TCR_normalized_counts  = sumOverRings(TCR_locations_array);
            pTCR_normalized_counts = sumOverRings(pTCR_probability_array);
        
            TCR_normalized_counts_array(n,:)  =  TCR_normalized_counts';
            pTCR_normalized_counts_array(n,:) =  pTCR_normalized_counts';
            
        end
        mean_TCR_normalized_counts_array = ...
            mean(TCR_normalized_counts_array,1);
        mean_pTCR_normalized_counts_array = ...
            mean(pTCR_normalized_counts_array,1);
        
        %%
        figure(6)
        bar(1:length(mean_TCR_normalized_counts_array),...
            mean_TCR_normalized_counts_array)
        hold on
        bar(1:length(mean_pTCR_normalized_counts_array),...
            10*mean_pTCR_normalized_counts_array)
        hold off
        
        dep_decay_TCR_normalized_counts(dep_ind,dec_ind,:) = ...
            mean_TCR_normalized_counts_array;
        dep_decay_pTCR_normalized_counts(dep_ind,dec_ind,:) = ...
            mean_pTCR_normalized_counts_array;
            
    end
end    














