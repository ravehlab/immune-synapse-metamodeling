array_size = 200;
array1  = zeros(array_size);
% array2  = zeros(array_size);

% array1(rand(array_size) < 0.01) = 1;
array2(rand(array_size) < 0.1) = 1;
array1(100,100) = 1;
% array1(:,:) = 1;
% array1(:,85:90) = 1;
% array1([1:95,105:200],:) = 0;
% spy(array1)

% array2(:,:) = 1;
% array2(:,[1:95,105:200]) = 0;
% array2([1:95,105:200],:) = 0;
% array2 = array1;
figure(3)
spy(array2)

gr = GR(array1,array2,R_max);
R = 0:R_max;

figure(2)
plot(R,gr,'.-')
hold on
plot([0 R_max],[1 1],'k--')
hold off
axis([0 R_max 0 5])
