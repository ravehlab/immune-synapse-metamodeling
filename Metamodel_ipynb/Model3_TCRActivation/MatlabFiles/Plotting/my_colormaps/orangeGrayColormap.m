function orange_gray_colormap = orangeGrayColormap(Nc)

reds   = 0:1/(Nc-1):1; 
greens = 0.5*reds; 
blues  = 0*ones(1,Nc); 

orange_gray_colormap = [reds',greens',blues'];

end