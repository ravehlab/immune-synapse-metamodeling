function parula_gray_colormap = parulaGrayColormap(Nc)

% reds   = 0:1/(Nc-1):1; 
% greens = 0.5*reds; 
% blues  = 0*ones(1,Nc); 

parula_gray_colormap = 0.8*parula(Nc);
parula_gray_colormap(1,:) = 0.9*[1 1 1];

end