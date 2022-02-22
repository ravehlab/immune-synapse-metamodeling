% results_of_runs_Barak_experimental
clear
clc

%{
doc:
return two 2-dimentional arrays of training data for model3.
X parameter is the decay_length of Lck* in nm.
Y parameter is the depletion_range between TCR and CD45 in nm.
For every training data value, distributions of TCR and CD45
point patterns are created. The TCRs are randomley scattered
inside a circle with diameter of 800nm with denstity of 
1000/um^2. The depletion range is an empty ring around
the TCR circle with width the depletion parameter. The remaining
area outside the empty ring is filled with randomley scatterd
CD45 points with density 1000/um^2. For every point distribution
pattern, we convolve the CD45 points with a radial distribution
function with an exponential decay of 'decay_length' in nm.
The resulting overall distribution marks the level of active 
Lck. The phosphorylation of TCRs points is random and is 
proportional to the level of Lck*. 
sub-functions: 

Simulation array size: 2um x 2um.
Number of pixels: 200 x 200.
%}
%% set plots parameters:
plots_parameters = setPlotsParameters();
%% set array sizes
array_size_x_patches = 200;
array_size_y_patches = 200;
A0 = zeros(array_size_x_patches,array_size_y_patches);
a = 10; % patch size nm
%
decay_lengths_nm = 10:10:200;
depletions_nm = 0:10:200;
phos_ratio_array = zeros(length(depletions_nm),...
                         length(decay_lengths_nm));
Rg_ratio_array = zeros(length(depletions_nm),...
                         length(decay_lengths_nm));                     
                     
for dep_ind = 11:11 %1:length(depletions_nm) %
    for dec_ind = 10:10 %1:length(decay_lengths_nm) %
        depletion_nm = depletions_nm(dep_ind);
        decay_length_nm = decay_lengths_nm(dec_ind);
        %% get tcr points locations
        % tcr parameters
        tcr_density = 1000; % #/um^2
        tcr_radius_patches = 40;

        tcr_array = tcrArray(A0,tcr_density,tcr_radius_patches);
        [tcr_x,tcr_y] = find(tcr_array);
        %
        %% get cd45 points locations
        % cd45 parameters:
        cd45_density = 1000; % #/um^2
        cd45_radius_patches = tcr_radius_patches + ...
            depletion_nm/a;

        cd45_array = cd45Array(A0,cd45_density,...
            cd45_radius_patches);
        [cd45_x,cd45_y] = find(cd45_array);
        %
        %% Lck distribution
        decay_length_patches = decay_length_nm/a;
        alck_array = aLckArray(...
            cd45_array,decay_length_patches);
        %
        %% tcr phosphorylation
        phos_threshold = 0.05; % 0.05
        
        ptcr_array = tcrPhosphorylation(...
            tcr_array,alck_array,phos_threshold);
        
        [ptcr_x,ptcr_y] = find(ptcr_array);
        %
        %% get radius of gyration
        phos_ratio = length(ptcr_x)/length(tcr_x);
        [Rg_ratio,tcr_r,ptcr_r] = phosAndRgRatio(...
            tcr_x,tcr_y,ptcr_x,ptcr_y);
        %
        %% create 'plot_data' struct:
        % data_to_plot.membrane_z = Z1;
        % data_to_plot.smoothed_Z = smoothed_Z;
        data_to_plot.Lck_array = alck_array;
        data_to_plot.tcr.x = tcr_x;
        data_to_plot.tcr.y = tcr_y;
        data_to_plot.tcr.r = tcr_r;
        data_to_plot.cd45.x = cd45_x;
        data_to_plot.cd45.y = cd45_y;
        data_to_plot.ptcr.x = ptcr_x;
        data_to_plot.ptcr.y = ptcr_y;
        data_to_plot.ptcr.r = ptcr_r;
        data_to_plot.decay_length_nm = decay_length_nm;
        data_to_plot.depletion_nm = depletion_nm;
        data_to_plot.phos_ratio = phos_ratio;
        data_to_plot.Rg_ratio = Rg_ratio;
        %
        %% data to arraye:
        phos_ratio_array(dep_ind,dec_ind) = phos_ratio;
        Rg_ratio_array(dep_ind,dec_ind) = Rg_ratio;
        %
        %% plot training data:
        plotTrainingDataModel3(plots_parameters,data_to_plot)
        %
    end
end
%
%%
[Decay_length_nm,Depletion_nm] = ...
    meshgrid(decay_lengths_nm,depletions_nm);

figure(2)
subplot(1,2,1)
h21 = surf(phos_ratio_array)';
h21.EdgeColor = 'none';
view(2)

axis equal
axis tight
colorbar

subplot(1,2,2)
h21 = surf(Rg_ratio_array)';
h21.EdgeColor = 'none';
view(2)

axis equal
axis tight
colorbar

%%
% writematrix(phos_ratio_array,'phos_ratio_array_randn.xls');
% writematrix(Rg_ratio_array,'Rg_ratio_array_randn.xls');
% dlmwrite('phos_ratio_array_randn.m',phos_ratio_array);
% dlmwrite('Rg_ratio_array_randn.m',Rg_ratio_array);



