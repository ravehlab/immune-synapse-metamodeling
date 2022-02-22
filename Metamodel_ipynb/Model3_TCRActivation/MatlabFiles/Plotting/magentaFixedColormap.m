function magenta_fixed_colormap = magentaFixedColormap(Nc)

reds   = 1.0*ones(1,Nc); 
greens = 0.3*ones(1,Nc);
blues  = 1.0*ones(1,Nc); 

magenta_fixed_colormap = [reds',greens',blues'];

end