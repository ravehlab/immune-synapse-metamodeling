function [] = plotActiveLckKernel(plots_parameters)

exponential_decay_kernel = ...
    ExponentialDecayKernel(100,10,1);


figure(71)
h = surf(exponential_decay_kernel);
h.EdgeColor = 'none';
colormap(plots_parameters.colormaps.magenta_fixed)
hold on
plot3(51,51,1,'r.','MarkerSize',20)
hold off
axis equal
grid off
box on
alpha color
alpha scaled
view(2)
% min_x = 0;
axis([0 100 0 100])
xticks([])
yticks([])

