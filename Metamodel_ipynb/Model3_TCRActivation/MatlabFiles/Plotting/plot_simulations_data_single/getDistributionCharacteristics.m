function [center_of_mass,moment_of_inertia] = ...
    getDistributionCharacteristics(x,y)

center_of_mass = [mean(x),mean(y)];
total_mass = length(x);

moment_of_inertia = (mean(x.^2) - center_of_mass(1)^2 +...
                     mean(y.^2) - center_of_mass(2)^2)/...
                     total_mass;

end