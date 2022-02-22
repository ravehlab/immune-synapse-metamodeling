function green_fixed_colormap = greenFixedColormap(Nc)

reds   = 0*ones(1,Nc); 
greens = 0.6*ones(1,Nc);
blues  = 0*ones(1,Nc); 

green_fixed_colormap = [reds',greens',blues'];

end