% multiple_colormaps
%{
https://www.mathworks.com/matlabcentral/answers/101346-how-do-i-use-multiple-colormaps-in-a-single-figure-in-r2014a-and-earlier
%}
array_size = size(Z1);
A0 = zeros(array_size);
linind_ptcr = sub2ind(size(A0),ptcr_x,ptcr_y);

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

%%
%{
ax = axis;
axis(ax)
Z = zeros(200);
% Project Data to the different planes.
h(1) = surf(Z'); 
h(2) = surf(smoothed_tcr);   
h(3) = surf(smoothed_cd45);
h(4) = surf(smoothed_ptcr);
% set(h,'FaceColor','interp','EdgeColor','interp')
% Build a colormap that consists of four separate
% colormaps.

cmap = [gray;tcr_colormap;cd45_colormap;ptcr_colormap];
colormap(cmap)

%}
%%
N_colors = 64;
TCR_color = [0.1 0.62 0.47];
CD45_color = [0.75 0.0 0.0];
pTCR_color = [0.9 0.67 0.0];

tcr_colormap = repmat(TCR_color,N_colors,1);
cd45_colormap = repmat(CD45_color,N_colors,1);
ptcr_colormap = repmat(pTCR_color,N_colors,1);
%%
%{
levels = 10:10:70;
figure(31)
h31 = surf(smoothed_Z');
hold on
contour3(smoothed_Z',levels,'LineColor','k')
hold off
colormap(gray)
colorbar
whitebg(1.0*[1 1 1])
h31.EdgeColor = 'none';
axis equal
axis tight
view(2)
% alpha color
% alpha scaled
grid off
xticks([0:50:200])
yticks([0:50:200])
xticklabels([])
yticklabels([])
box on
caxis([0 70])
%

figure(32)
h32 = surf(smoothed_tcr);
colormap(tcr_colormap)
whitebg(0.4*[1 1 1])
h32.EdgeColor = 'none';
axis equal
axis tight
view(2)
alpha color
alpha scaled
grid off
xticks([0:50:200])
yticks([0:50:200])
xticklabels([])
yticklabels([])
box on
caxis([0 70])
%
%%
figure(33)
h33 = surf(smoothed_cd45);
colormap(cd45_colormap)
whitebg(0.4*[1 1 1])
h33.EdgeColor = 'none';
axis equal
axis tight
view(2)
alpha color
alpha scaled
grid off
xticks([0:50:200])
yticks([0:50:200])
xticklabels([])
yticklabels([])
box on
caxis([0 70])

%%
figure(34)
h34 = surf(smoothed_ptcr);
colormap(ptcr_colormap)
% whitebg(0.4*[1 1 1])
h34.EdgeColor = 'none';
axis equal
axis tight
view(2)
alpha color
alpha scaled
grid off
xticks([0:50:200])
yticks([0:50:200])
xticklabels([])
yticklabels([])
box on
caxis([0 70])

%}
%% Create four axes
figure(40)
clf
f = 0.2;
levels = 10:10:70;
% subplot(1,2,1)
% whitebg('w')

% ax1
ax1 = axes;
h1 = surf(ax1,smoothed_Z');
hold on
contour3(smoothed_Z',levels,'LineColor',[0 0 0])
hold off
h1.EdgeColor = 'none';
view(2)
axis equal
axis tight
axis([f*array_size(1), (1-f)*array_size(1),...
      f*array_size(2), (1-f)*array_size(2)])

% ax2
ax2 = axes;
h2 = surf(ax2,smoothed_tcr);
h2.EdgeColor = 'none';
view(2)
alpha color
alpha scaled
axis equal
axis tight
axis([f*array_size(1), (1-f)*array_size(1),...
      f*array_size(2), (1-f)*array_size(2)])

% ax3
ax3 = axes;
h3 = surf(ax3,smoothed_cd45);
h3.EdgeColor = 'none';
view(2)
alpha color
alpha scaled
axis equal
axis tight
axis([f*array_size(1), (1-f)*array_size(1),...
      f*array_size(2), (1-f)*array_size(2)])

% ax4 
ax4 = axes;
h4 = surf(ax4,smoothed_ptcr);
h4.EdgeColor = 'none';
view(2)
alpha color
alpha scaled
axis equal
axis tight
axis([f*array_size(1), (1-f)*array_size(1),...
      f*array_size(2), (1-f)*array_size(2)])

% plot([1 200],[1,200],'b-')

ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];

% Hide the top axes
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];

ax3.Visible = 'off';
ax3.XTick = [];
ax3.YTick = [];

% Hide the top axes
ax4.Visible = 'off';
ax4.XTick = [];
ax4.YTick = [];
% axis([f*array_size(1), (1-f)*array_size(1),...
%       f*array_size(2), (1-f)*array_size(2)])
% axis equal
% axis tight
% Link them together
linkaxes([ax1,ax2,ax3,ax4])

%%Give each one its own colormap
colormap(ax1,gray)
colormap(ax2,tcr_colormap)
colormap(ax3,cd45_colormap)
colormap(ax4,ptcr_colormap)

% Then add colorbars and get everything lined up
% set([ax1,ax2,ax3,ax4],'Position',[0.1 0.10 0.8 0.815]);

% set([ax1,ax2,ax3,ax4],'Position',[0.17 0.11 0.685 0.815]);
% cb1 = colorbar(ax1,'Position',[.05 .11 .0675 .815]);
% cb2 = colorbar(ax2,'Position',[.88 .11 .0675 .815]);

axis([f*array_size(1), (1-f)*array_size(1),...
      f*array_size(2), (1-f)*array_size(2)])

% axis equal
% axis tight
axis off
%%
diag_z = diag(smoothed_Z');
diag_cd45 = diag(smoothed_cd45');
diag_tcr = diag(smoothed_tcr');
diag_ptcr = diag(smoothed_ptcr');

lw = 2;

figure(41)
clf
% subplot(1,2,2)

% plot(diag_z/1000,'Color',0.4*[1 1 1],'LineWidth',lw)
max_diag_z = max(diag_z);
hold on
for l = 1:length(diag_z)-1
    plot([l-1,l],[diag_z(l),diag_z(l+1)]/1000,...
        'Color',diag_z(l)*[1 1 1]/max_diag_z,...
        'LineWidth',2*lw)
end
hold off

hold on
plot(diag_cd45,'Color',CD45_color,'LineWidth',lw)
plot(diag_tcr,'Color',TCR_color,'LineWidth',lw)
plot(diag_ptcr,'Color',pTCR_color,'LineWidth',lw)

hold off
axis([f*array_size(1), (1-f)*array_size(1), 0, 0.12])
box on

% XTick = 0:50:200;
% YTick = 0:50:200;

