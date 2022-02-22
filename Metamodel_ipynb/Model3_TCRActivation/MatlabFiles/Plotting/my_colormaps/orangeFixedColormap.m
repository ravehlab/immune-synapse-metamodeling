function orange_fixed_colormap = orangeFixedColormap(Nc)

reds   = 1*ones(1,Nc); 
greens = 0.5*reds; 
blues  = 0*ones(1,Nc); 

orange_fixed_colormap = [reds',greens',blues'];

end