uiopen('C:\Users\yairn\Git\immune-synapse-metamodeling\Model3 TCRActivation\MatlabFiles\Plotting\zap70_map1.PNG',1);
% [Z map] = rgb2ind( cdata, 256 );
[Z map] = rgb2ind(cdata, parula);

%%
figure(5)
imagesc(Z)


%%
zap70_color = [0.0 0.5 1.0];
pTCR_color = [1.0 0.5 0.0];
t_sec = 30; %75; %30;
t_ind = round((2/3)*t_sec);

% h = fspecial('gaussian',11,3); % 1
h = fspecial('gaussian',15,5);
Z2 = imfilter(Z,h,'replicate'); 

figure(6)
imagesc(Z2)
axis equal
axis tight
axis([0 160 0 220])
hold on
plot([t_ind, t_ind],[0 220], '-','Color',zap70_color, 'LineWidth',2)
hold off

xlabel('Time [sec]')
xticks([0 100 200 300 400]/3)
xticklabels([0 50 100 150 200])

ylabel('Distance [nm]')
yticks([0 55 110 165 220])
yticklabels([2000 1000 0 -1000 -2000])

%% Cross section plot:
max_x = 220;
Nx = 4;
cross_section_x = 1:max_x;
cross_section_y = flipud(Z2(1:max_x,t_ind));
figure(7)
plot(cross_section_x, cross_section_y, '-','Color',zap70_color, 'LineWidth',2)
xlabel('Distance [nm]')

xticks([0 1*max_x/Nx 2*max_x/Nx 3*max_x/Nx max_x])
xticklabels([-2000 -1000 0 1000 2000])
axis([0 max_x 0 255])

%%% get valid y values:
% Cell center location:
start_index = 50;
% Cell edge location:
end_index = 110;

x_mask = ones(1,max_x);
x_mask(1:start_index) = 0;
x_mask(end_index:end) = 0;

cross_section_y = double(cross_section_y);
valid_y = cross_section_y.*x_mask';

figure(7)
hold on
plot(cross_section_x, valid_y, ...
    '-','Color',pTCR_color, 'LineWidth',2)
hold off


%%% rMean
masked_valid_y = cross_section_x.*valid_y';
rMean_pTCR0 = sum(masked_valid_y)/sum(valid_y);
rMean_pTCR_ratio = rMean_pTCR0/end_index;

figure(7)
hold on
plot(rMean_pTCR0, 5, ...
    'v','MarkerFaceColor',0.3*[1.0, 1.0, 1.0],...
    'Color',0.3*[1.0, 1.0, 1.0],...
    'MarkerSize',8)


hold off

%%

cropped_valid_y = masked_valid_y;
cropped_valid_y(cropped_valid_y == 0) = [];
cropped_valid_x = 1:length(cropped_valid_y);

rMean_pTCR = sum(cropped_valid_y.*cropped_valid_x)/sum(cropped_valid_y);
rMean_pTCR_ratio = rMean_pTCR/length(cropped_valid_y);
norm_cropped_valid_y = cropped_valid_y/sum(cropped_valid_y);

figure(8)
plot(cropped_valid_x, norm_cropped_valid_y, ...
    '-','Color',pTCR_color, 'LineWidth',2)

hold on
plot(rMean_pTCR, 0, ...
    'v','MarkerFaceColor',0.3*[1.0, 1.0, 1.0],...
    'Color',0.3*[1.0, 1.0, 1.0],...
    'MarkerSize',8)
hold off

%%
% Z3 = fliplr(flipud(Z2'));
Z3 = rot90(rot90(Z2'));
figure(26)
imagesc(Z3)
axis equal
axis tight
axis([0 220 0 160])
hold on
plot([0 220],[160-t_ind, 160-t_ind], '-','Color',pTCR_color, 'LineWidth',2)
hold off

xlabel('Distance [nm]')
xticks([0 55 110 165 220])
xticklabels([-2000 -1000 0 1000 2000])

ylabel('Time [sec]')
yticks(160- [400 300 200 100 0]/3)
yticklabels([200 150 100 50 0])

%%
figure(27)
% hold on
% plot(cross_section_x, cross_section_y, ...
%     '-','Color',pTCR_color, 'LineWidth',2)
bar(cross_section_x, cross_section_y, 'BarWidth',1.0, ...
    'FaceColor',pTCR_color)
axis([0 220 0 160])
xlabel('Distance [nm]')
xticks([0 55 110 165 220])
xticklabels([-2000 -1000 0 1000 2000])
% hold off
%%
figure(37)
clf
patch([cross_section_x(1),cross_section_x,cross_section_x(end)],...
      [0,cross_section_y',0], pTCR_color,...
      'EdgeColor','none')
  
axis([0 220 0 160])
xlabel('Distance [nm]')
xticks([0 55 110 165 220])
xticklabels([-2000 -1000 0 1000 2000])
%%
figure(28)
bar(cropped_valid_x, norm_cropped_valid_y,'BarWidth',1, ...
    'FaceColor',pTCR_color,'EdgeColor','none')

% hold on
% plot(rMean_pTCR, 0, ...
%     'v','MarkerFaceColor',0.3*[1.0, 1.0, 1.0],...
%     'Color',0.3*[1.0, 1.0, 1.0],...
%     'MarkerSize',8)
% hold off

figure(38)
patch([cropped_valid_x(1),cropped_valid_x,cropped_valid_x(end)],...
      [0,norm_cropped_valid_y,0],...
      pTCR_color,'EdgeColor','none')

