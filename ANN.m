%% Lecture 08
% Andy Worthington
tic
clear all; close all; clc;

%% Training Image
% Import and Pre-Process Training Image
Igray = imread('training.jpg');
BW = ~im2bw(Igray);

SE = strel('disk',1);
BW2 = imerode(BW, SE);
imshow(BW2)


labels = bwlabel(BW2);
Iprops = regionprops(labels);

Iprops( [Iprops.Area] < 250 ) = [];
num = length( Iprops );

Ibox = floor( [Iprops.BoundingBox] );
Ibox = reshape(Ibox,[4 num]);

% imshow(Igray,'border','tight');
%
% hold on;
% for k = 1:num
%     rectangle('position',Ibox(:,k),'edgecolor','g','LineWidth',1);
%
%     col = Ibox(1,k);
%     row = Ibox(2,k);
%
%     text(col,row-50,sprintf('%2.2d',k), ...
%         'fontsize',16,'color','r','fontweight','bold');
% end


for k = 1:num
    col1 = Ibox(1,k);
    col2 = Ibox(1,k) + Ibox(3,k);
    row1 = Ibox(2,k);
    row2 = Ibox(2,k) + Ibox(4,k);
    subImage = BW2(row1:row2, col1:col2);
    
    subImageScaled = imresize(subImage, [24 12]);
    
    TPattern(k,:) = subImageScaled(:)';
end
figure,imshow(subImage);
figure,imshow(subImageScaled);
figure,imshow(TPattern);
%TTarget = eye(10)';

TTarget = zeros(100,10);

for row = 1:10
    for col = 1:10
        TTarget( 10*(row-1) + col, row ) = 1;
    end
end

TPattern = TPattern';
TTarget = TTarget';

% Train ANN from Training Image
mynet = newff([zeros(288,1) ones(288,1)], [24 10], {'logsig' 'logsig'}, 'traingdx');
mynet.trainParam.epochs = 500;
mynet = train(mynet,TPattern,TTarget);


% mynet = feedforwardnet(24,'traingdx');
% mynet.trainParam.epochs = 500;
% mynet = train(mynet,TPattern,TTarget)

%% Unknown Images (Handwriting)

% Import and Pre-Process Unknown Images
imstr = [196128; 480000; 480096; 603032];
imarray =   [1 9 6 1 2 8;
            4 8 0 0 0 0;
            4 8 0 0 9 6;
            6 0 3 0 3 2];
for h = 1:length(imstr)
    Igray = imread(sprintf('%6d.jpg', imstr(h)));
%      Igray = imread('196128.jpg');
    BW = ~im2bw(Igray);
    
    SE = strel('disk',1);
    BW2 = imerode(BW, SE);
    
    labels = bwlabel(BW2);
    Iprops = regionprops(labels);
    
    Iprops( [Iprops.Area] < 1000 ) = [];
    num = length( Iprops );
    
    Ibox = floor( [Iprops.BoundingBox] );
    Ibox = reshape(Ibox,[4 num]);
    
    
    for b = 1:num
        col1 = Ibox(1,b);
        col2 = Ibox(1,b) + Ibox(3,b);
        row1 = Ibox(2,b);
        row2 = Ibox(2,b) + Ibox(4,b);
        
        subImage = BW2(row1:row2, col1:col2);
        subImageScaled = imresize(subImage, [24 12]);
        UPattern(b,:) = subImageScaled(:)';
    end
    
    UPattern = UPattern';
    Y = sim(mynet, UPattern);
    [values, index] = max(Y(:,:));
    totalNumbers(h, :) = (index - 1);
    toc
    UPattern = 0;
    clear UPattern;
end
% Done, just add table
totalNumbers