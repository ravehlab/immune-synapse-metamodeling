function [] = plotCross_sections_Z_TCR_CD45_pTCR(...
    plots_parameters,Z1,...
    tcr_x,tcr_y,cd45_x,cd45_y,ptcr_x,ptcr_y,alpha)

linind_tcr = sub2ind(size(Z1),tcr_x,tcr_y);
linind_cd45 = sub2ind(size(Z1),cd45_x,cd45_y);
linind_ptcr = sub2ind(size(Z1),ptcr_x,ptcr_y);

array_size = size(Z1);
A0 = zeros(array_size);

smoothed_Z = imgaussfilt(Z1,8);

A_tcr = A0;
A_tcr(linind_tcr) = 1;
smoothed_tcr = imgaussfilt(A_tcr,8);

A_cd45 = A0;
A_cd45(linind_cd45) = 1;
smoothed_cd45 = imgaussfilt(A_cd45,8);

A_ptcr = A0;
A_ptcr(linind_ptcr) = 1;
smoothed_ptcr = imgaussfilt(A_ptcr,8);


diag_z = diag(smoothed_Z');
diag_cd45 = diag((smoothed_cd45'));
diag_tcr = diag(smoothed_tcr');
diag_ptcr = diag(smoothed_ptcr');



f = 0.2;
lw = 3.0;

max_diag_z = max(diag_z);
if 0
    hold on
    for l = 1:length(diag_z)-1
        plot([l-1,l],[diag_z(l),diag_z(l+1)]/1000,...
            'Color',diag_z(l)*[1 1 1]/max_diag_z,...
            'LineWidth',2*lw)
    end
    hold off
end
if 0
    hold on
        plot(diag_cd45,'Color',plots_parameters.cd45.color,...
            'LineWidth',lw)
    hold off
end

if 1
    hold on
    plot(diag_tcr,'Color',plots_parameters.tcr.color,...
        'LineWidth',lw)
    hold off
end

if 1
    hold on
    plot(diag_ptcr,'Color',plots_parameters.ptcr.color,...
        'LineWidth',lw)

    hold off
end
axis([f*array_size(1), (1-f)*array_size(1), 0, 0.15])

xticks([])
yticks([])
box on
drawnow

end