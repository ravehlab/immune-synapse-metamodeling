% bar_Z_tcr_cd45_ptcr_hist

TCR_color = [0.1 0.62 0.47];
CD45_color = [0.75 0.0 0.0];
pTCR_color = [0.9 0.67 0.0];

linind_ptcr = sub2ind(size(Z1),ptcr_x,ptcr_y);


figure(61)
h61 = surf(Z1');
colormap(gray)
h61.EdgeColor = 'none';
view(2)
axis equal
axis tight

hold on
scatter3(tcr_x,tcr_y,100*ones(size(tcr_x)),10,'fill',...
    'MarkerFaceColor',TCR_color)
hold off

hold on
scatter3(cd45_x,cd45_y,100*ones(size(cd45_x)),10,'fill',...
    'MarkerFaceColor',CD45_color)
hold off

hold on
scatter3(ptcr_x,ptcr_y,100*ones(size(ptcr_x)),10,'fill',...
    'MarkerFaceColor',pTCR_color)
hold off

xticks([0:50:200])
yticks([0:50:200])
% xticklabels([])
% yticklabels([])
box on
caxis([0 70])

%% Z histogram
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

face_alpha = 1.0;
figure(62)
clf

%
hold on
for m = 1:length(edges)-1
    
    bar(m,norm2_N_z(m),1,'FaceColor',0.1+m/90*[1 1 1],...
        'FaceAlpha',0.5,...
    'EdgeColor',0.1+m/90*[1 1 1],...
    'LineWidth',1.0);
end
hold off

hold on
bar(edges(2:end),norm2_N_cd45,1,...
    'FaceColor',CD45_color,...
    'FaceAlpha',face_alpha,...
    'EdgeColor','none')

bar(edges(2:end),norm2_N_tcr,1,...
    'FaceColor',TCR_color,...
    'FaceAlpha',face_alpha,...
    'EdgeColor','none')

bar(edges(2:end),norm2_N_ptcr,1,...
    'FaceColor',pTCR_color,...
    'FaceAlpha',face_alpha,...
    'EdgeColor','none')

hold off

ylim([0 1])


