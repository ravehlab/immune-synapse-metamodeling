function [] = plotTCR_CD45_Z_2D(tcr_r,tcr_z,cd45_r,cd45_z,Z)

figure(61)
clf
hold on
for tcr_i = 1:length(tcr_r)
    h_tcr = plot(tcr_r(tcr_i)*[1, 1],tcr_z(tcr_i)*[0, 1],...
        '-g','LineWidth',2); %
    h_tcr.Color(4) = 1.0;
end
% mirror
for tcr_i = 1:length(tcr_r)
    h_tcr = plot(-tcr_r(tcr_i)*[1, 1],tcr_z(tcr_i)*[0, 1],...
        '-g','LineWidth',2); %
    h_tcr.Color(4) = 1.0;
end

for cd45_i = 1:length(cd45_r)
    h_cd45 = plot(cd45_r(cd45_i)*[1, 1],cd45_z(cd45_i)*[0, 1],...
        '-r','LineWidth',2); %
    h_cd45.Color(4) = 0.1;

end
% mirror
for cd45_i = 1:length(cd45_r)
    h_cd45 = plot(-cd45_r(cd45_i)*[1, 1],cd45_z(cd45_i)*[0, 1],...
        '-r','LineWidth',2); %
    h_cd45.Color(4) = 0.1;

end
hold off
axis([-100 100 0 80])

end