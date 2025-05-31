% Organize the 17flowers Dataset into flower-named folders
originalDataset = fullfile(pwd, '17flowers');

%the organised dataset with the file names
organisedDataset = fullfile(pwd, '17flowers_named');
if ~exist(organisedDataset, 'dir')
    mkdir(organisedDataset);
end

% Flower class names 
flowerNames = {'Daffodil', 'Snowdrop','LilyValley','Bluebell','Crocus','Iris','Tigerlily',...
    'Tulip','Fritillary','Sunflower','Daisy','ColtsFoot',...
    'Dandelion','Cowslip','Buttercup','Windflower','Pansy'};

%save each flower into the corresponding folder
for classIdx = 1:17
    flowerName = flowerNames{classIdx};
    classFolderPath = fullfile(organisedDataset, flowerName);
    
    if ~exist(classFolderPath, 'dir')
        mkdir(classFolderPath);
    end

    for imgIdx = 1:80
        globalIdx = (classIdx - 1) * 80 + imgIdx;
        imgName = sprintf('image_%04d.jpg', globalIdx);
        srcPath = fullfile(originalDataset, imgName);
        dstPath = fullfile(classFolderPath, imgName);
        
        if exist(srcPath, 'file')
            copyfile(srcPath, dstPath);
        else
            warning('Image not found: %s', srcPath);
        end
    end
end


imds = imageDatastore('17flowers_named', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Resize all images to 256x256
imds.ReadFcn = @(filename) imresize(imread(filename), [256 256]);

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXShear', [-10 10], ...
    'RandYShear', [-10 10], ...
    'RandXReflection', true, ...
    'RandScale', [0.9 1.1]);

augImdsTrain = augmentedImageDatastore([256 256 3], imdsTrain, ...
    'DataAugmentation', imageAugmenter);


layers = [
    imageInputLayer([256 256 3], 'Name', 'input')

    convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv_1')
    batchNormalizationLayer('Name', 'bn_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'bn_2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'bn_3')
    reluLayer('Name', 'relu_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_3')

    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv_4')
    batchNormalizationLayer('Name', 'bn_4')
    reluLayer('Name', 'relu_4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_4')

    fullyConnectedLayer(512, 'Name', 'fc_1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.5, 'Name', 'dropout1')

    fullyConnectedLayer(128, 'Name', 'fc_2')
    reluLayer('Name', 'relu_fc2')
    dropoutLayer(0.4, 'Name', 'dropout2')

    fullyConnectedLayer(17, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];



options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 18, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

classnet = trainNetwork(augImdsTrain, layers, options);

YPred = classify(classnet, imdsValidation);
YTrue = imdsValidation.Labels;

accuracy = mean(YPred == YTrue);
confusionchart(YTrue, YPred);

save('classnet.mat', 'classnet');

fprintf("Validation Accuracy: %.2f%%\n", accuracy * 100);



