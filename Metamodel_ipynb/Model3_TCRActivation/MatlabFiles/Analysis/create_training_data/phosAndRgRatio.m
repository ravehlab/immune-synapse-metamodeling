function [Rg_ratio,tcr_r,ptcr_r] = phosAndRgRatio(...
    tcr_x,tcr_y,ptcr_x,ptcr_y)

% center of mass: 
mean_tcr_x = mean(tcr_x);
mean_tcr_y = mean(tcr_y);
N_tcr = length(tcr_x);
N_ptcr = length(ptcr_x);

% tcr:
% Distance, (r), from center of mass:
r_tcr_squared = ((tcr_x - mean_tcr_x).^2 +...
                 (tcr_y - mean_tcr_y).^2);
tcr_r = sqrt(r_tcr_squared);
% Radius of gyration:
Rg_tcr = sqrt(sum(r_tcr_squared./N_tcr));

% ptcr:
% Distance, (r), from center of mass:
r_ptcr_squared = ((ptcr_x - mean_tcr_x).^2 +...
                  (ptcr_y - mean_tcr_y).^2);
ptcr_r = sqrt(r_ptcr_squared);
% Radius of gyration:
Rg_ptcr = sqrt(sum(r_ptcr_squared./N_ptcr));

% Radius of gyration ratio:
Rg_ratio = Rg_ptcr/Rg_tcr;

end