digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imds.ReadSize = 500;
rng(0)
imds = shuffle(imds);
[imdsTrain,imdsVal,imdsTest] = splitEachLabel(imds,0.95,0.025);
dsTestNoisy = transform(imdsTest,@addNoise);
dsTest = combine(dsTestNoisy,imdsTest);
dsTest = transform(dsTest,@commonPreprocessing);
timg_all=read(dsTest);
load nnet.mat
ypred = predict(net,dsTest);
%% displaying 10 test imgs, ssim, rmse vals
trg=52:61;
ytest=ypred(:,:,:,trg);
ytest=reshape(ytest,[32,32,10]);
ycell=num2cell(ytest,[1,2]);
ycell=reshape(ycell,[10,1]);

timg=timg_all(trg,:);
minibatch = cat(3,timg(:,2),timg(:,1),ycell);
figure;
montage(permute(minibatch,[3 2 1]),'Size',[10 3])
title('original, noisy, denoised');
rmse_val = zeros(size(trg));
ssim_val=zeros(size(trg));
for i=1:10
    rmse_val(i)=norm(ycell{i}-timg{i,2})/norm(timg{i,2});
    ssim_val(i)=ssim(ycell{i}, timg{i,2});
end

disp('rmse : ');
disp(rmse_val);
disp('ssim: ');
disp(ssim_val);
%% displaying collage of filters in each layer
lyrs={'conv1', 'conv2', 'conv3', 'tconv1', 'tconv2', 'tconv3', 'dec_conv1'};
num_filts=[16,32,64,32,16,8,1];

for i = 1:numel(lyrs)
    I = deepDreamImage(net,lyrs{i},1:num_filts(i),'Verbose',false,'PyramidLevels',1);
    figure;
    I = imtile(I,'ThumbnailSize',[64 64]);imshow(I);
    str=[lyrs{i},' layer filters'];
    title(str);
end

%% displaying activations
grid_sz={[4,4], [8,4], [8,8], [8,4], [4,4], [4,2], [1,1] };
im=timg{4,1};
for i = 1:numel(lyrs)
    act1 = activations(net,im,lyrs{i});
    sz = size(act1);
    if (i<numel(lyrs))
        act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
    end
    I = imtile(mat2gray(act1),'GridSize',grid_sz{i},'ThumbnailSize',[64 64]);
    figure;
    imshow(I);
    str=[lyrs{i},' layer activations'];
    title(str);
end
