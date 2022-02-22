function [] = plotBarZ_TCR_CD45_pTCR(plots_parameters,Z1,...
    tcr_x,tcr_y,cd45_x,cd45_y,ptcr_x,ptcr_y,alpha)

linind_tcr = sub2ind(size(Z1),tcr_x,tcr_y);
linind_cd45 = sub2ind(size(Z1),cd45_x,cd45_y);
linind_ptcr = sub2ind(size(Z1),ptcr_x,ptcr_y);

% Z histogram
Z_z = Z1(:);
tcr_z = Z1(linind_tcr);
cd45_z = Z1(linind_cd45);
ptcr_z = Z1(linind_ptcr);

edges = 0:1:80;
[N_z,~] = histcounts(Z_z,edges);
[N_tcr,~] = histcounts(tcr_z,edges);
[N_cd45,~] = histcounts(cd45_z,edges);
[N_ptcr,~] = histcounts(ptcr_z,edges);

norm_N_z = N_z/sum(N_z);
norm_N_tcr = N_tcr/sum(N_tcr);
norm_N_cd45 = N_cd45/sum(N_cd45);
norm_N_ptcr = N_ptcr/sum(N_tcr); %!!!

max_h = 0.9;
norm2_N_z = max_h*norm_N_z/max(norm_N_z);
norm2_N_tcr = max_h*norm_N_tcr/max(norm_N_tcr);
norm2_N_cd45 = max_h*norm_N_cd45/max(norm_N_cd45);
norm2_N_ptcr = max_h*norm_N_ptcr/max(norm_N_tcr); %!!!

face_alpha = alpha;

hold on
for m = 1:length(edges)-1   
    bar(m,norm2_N_z(m),1,'FaceColor',0.1+m/90*[1 1 1],...
    'EdgeColor','none',...
    'LineWidth',1.0);
end
hold off

hold on
bar(edges(2:end),norm2_N_cd45,1,...
    'FaceColor',plots_parameters.CD45.color,...
    'FaceAlpha',face_alpha,...
    'EdgeColor','none')

bar(edges(2:end),norm2_N_tcr,1,...
    'FaceColor',plots_parameters.TCR.color,...
    'FaceAlpha',face_alpha,...
    'EdgeColor','none')

bar(edges(2:end),norm2_N_ptcr,1,...
    'FaceColor',plots_parameters.pTCR.color,...
    'FaceAlpha',face_alpha,...
    'EdgeColor','none')
hold off

ylim([0 1])

xticks([])
yticks([])
% xticklabels(10*xticks)
% yticklabels(10*yticks)
drawnow

%}
end