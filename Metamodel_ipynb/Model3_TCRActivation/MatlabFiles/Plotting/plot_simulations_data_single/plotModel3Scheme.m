function [] = plotModel3Scheme(array_size_nm,tcr_density,...
    phosphorylation_probability,plots_parameters)

array_size_microns = array_size_nm/1000;
array_area_microns = array_size_microns^2;
N_tcr = tcr_density*array_area_microns;

tcr_x = array_size_nm*rand(N_tcr,1);
tcr_y = array_size_nm*rand(N_tcr,1);

N_ptcr = round(N_tcr*phosphorylation_probability);

rand_choice = randsample(N_tcr,N_ptcr,false);

ptcr_x = tcr_x(rand_choice);
ptcr_y = tcr_y(rand_choice);

Lck_array = 0.1*ones(array_size_nm);

figure(72)
h = surf(Lck_array);
h.EdgeColor = 'none';
colormap(plots_parameters.colormaps.magenta_fixed)
h.FaceAlpha = 0.1;
axis equal
grid off
box on
% alpha color
% alpha scaled

hold on
plot3(tcr_x,tcr_y,ones(size(tcr_x)),...
    '.','Color',plots_parameters.TCR.color,...
    'MarkerSize',10)
plot3(ptcr_x,ptcr_y,ones(size(ptcr_x)),...
    '.','Color',[1.0, 0.65, 0.0],...
    'MarkerSize',10)
hold off

view(2)
axis([0 array_size_nm 0 array_size_nm])
xticks([])
yticks([])

end