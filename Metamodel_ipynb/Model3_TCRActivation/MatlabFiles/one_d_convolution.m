% one_d_convlution
sigma = 500;
dx = 10; %sigma/10;
gaussian_x = -4*sigma:dx:4*sigma;
gaussian_y = exp(-0.5*((gaussian_x)/sigma).^2);

%%
size_x = 1200; % nm
gap = 100; % nm
%% TCR (1)
length_square1 = 400;
start1 = size_x/2 - length_square1/2;
end1 = size_x/2 + length_square1/2;
square1_x = 0:1:size_x;
square1_y = zeros(size(square1_x));
square1_y(start1:1:end1) = 1;

%% CD45 (2)
length_square2 = 250;
start2_left = start1 - gap - length_square2;
end2_left = start2_left + length_square2;
square2_x = 0:1:size_x;
square2_y = zeros(size(square2_x));
square2_y(start2_left:1:end2_left) = 1;


start2_right = end1 + gap;
end2_right = start2_right + length_square2;
square2_x = 0:1:size_x;
square2_y(start2_right:1:end2_right) = 1;
%%
conv1 = conv(square1_y,gaussian_y,'same');
conv2_left = conv(square2_y,gaussian_y,'same');
%%
figure(9)
clf
subplot(3,1,1)
plot(gaussian_x,gaussian_y,'k-')

subplot(3,1,2)
plot(square1_x,square1_y,'r-')
hold on
plot(square2_x,square2_y,'g-')
hold off


subplot(3,1,3)
plot(conv1,'r-')
hold on
plot(conv2_left,'g-')
hold off
xlim([0 1200])




