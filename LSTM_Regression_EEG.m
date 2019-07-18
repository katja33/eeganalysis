%% Regress the EEG time series data using Deep Learning (LSTM network)
% This example shows how to regress the EEG time series data using the
% recurrent neural network. Please prepare the EEG sample data
% (Subject1_2D.mat) in advance to run this script. You can find it from the
% Link below -> Dataset 2 2D motion section -> [DOWNLOAD]
% <https://sites.google.com/site/projectbci/>
%  
% SubjectÅF21 year old, male, right handed, no known medical conditions
% EEG: 19 electrodes (FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ 
% PZ)
% Data acquired by Neurofax EEG system, exported by Eemagine EEG
% Sampling rate: 500 Hz
%
% This script regresses the 0.1 second (50 samples) future EEG time series data.
% Please note that the network is not well converged.

%% Loading the data
load Subject1_2D.mat

%% Prepare the training data
for ii = 1:59
    XTrain{ii} = LeftBackward1(50*(ii-1)+1:50*ii,:)';
end
for ii = 1:59
    XTrain{ii+59} = LeftBackward2(50*(ii-1)+1:50*ii,:)';
end
for ii = 1:59
    XTrain{ii+118} = LeftForward1(50*(ii-1)+1:50*ii,:)';
end
for ii = 1:59
    XTrain{ii+177} = LeftForward2(50*(ii-1)+1:50*ii,:)';
end
for ii = 1:59
    XTrain{ii+236} = RightBackward1(50*(ii-1)+1:50*ii,:)';
end
for ii = 1:59
    XTrain{ii+295} = RightBackward2(50*(ii-1)+1:50*ii,:)';
end
for ii = 1:59
    XTrain{ii+354} = RightForward1(50*(ii-1)+1:50*ii,:)';
end
for ii = 1:59
    XTrain{ii+413} = RightForward2(50*(ii-1)+1:50*ii,:)';
end
XTrain = XTrain';

for ii = 2:60
    YTrain{ii-1} = LeftBackward1(50*(ii-1)+1:50*ii,:)';
end
for ii = 2:60
    YTrain{ii+58} = LeftBackward2(50*(ii-1)+1:50*ii,:)';
end
for ii = 2:60
    YTrain{ii+117} = LeftForward1(50*(ii-1)+1:50*ii,:)';
end
for ii = 2:60
    YTrain{ii+176} = LeftForward2(50*(ii-1)+1:50*ii,:)';
end
for ii = 2:60
    YTrain{ii+235} = RightBackward1(50*(ii-1)+1:50*ii,:)';
end
for ii = 2:60
    YTrain{ii+294} = RightBackward2(50*(ii-1)+1:50*ii,:)';
end
for ii = 2:60
    YTrain{ii+353} = RightForward1(50*(ii-1)+1:50*ii,:)';
end
for ii = 2:60
    YTrain{ii+412} = RightForward2(50*(ii-1)+1:50*ii,:)';
end
YTrain = YTrain';

%% Layer settings
% inputSize = 19;
% numHiddenUnits = 100;
% numResponses = 2;

numResponses = size(YTrain{1},1);
featureDimension = size(XTrain{1},1);
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%% Option settings
maxEpochs = 100;
miniBatchSize = 32;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);

%% Train the data
net = trainNetwork(XTrain,YTrain,layers,options);

%% Prepare the test data
for ii = 1:59
    XTest{ii} = LeftBackward3(50*(ii-1)+1:50*ii,:)';
end
for ii = 1:59
    XTest{ii+59} = LeftForward3(50*(ii-1)+1:50*ii,:)';
end
for ii = 1:59
    XTest{ii+118} = RightBackward3(50*(ii-1)+1:50*ii,:)';
end
for ii = 1:59
    XTest{ii+177} = RightForward3(50*(ii-1)+1:50*ii,:)';
end
XTest = XTest';

for ii = 2:60
    YTest{ii-1} = LeftBackward3(50*(ii-1)+1:50*ii,:)';
end
for ii = 2:60
    YTest{ii+58} = LeftForward3(50*(ii-1)+1:50*ii,:)';
end
for ii = 2:60
    YTest{ii+117} = RightBackward3(50*(ii-1)+1:50*ii,:)';
end
for ii = 2:60
    YTest{ii+176} = RightForward3(50*(ii-1)+1:50*ii,:)';
end
YTest = YTest';

%% Validate the data
YPredict = predict(net, XTest);

%% Visualize the accuracy
figure;
plot(YTest{1}(1,:), '-o');
hold on
plot(YPredict{1}(1,:), '-*');
legend('Original', 'Predicted');