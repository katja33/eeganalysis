%% Classify the EEG time series data using Deep Learning (LSTM network)
% This example shows how to classify the EEG time series data using the
% recurrent neural network. Please prepare the EEG sample data
% (Subject1_2D.mat) in advance to run this script. You can find it from the
% Link below -> Dataset 2 2D motion section -> [DOWNLOAD]
% <https://sites.google.com/site/projectbci/ https://sites.google.com/site/projectbci/>
%  
% SubjectÅF21 year old, male, right handed, no known medical conditions
% EEG: 19 electrodes (FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ 
% PZ)
% Data acquired by Neurofax EEG system, exported by Eemagine EEG
% Sampling rate: 500 Hz
%
% This script classifies the EEG time series data between the movement of LeftBackward and RightForward.

%% Loading the data
load Subject1_2D.mat

%% Layer settings
inputSize = 19;
numHiddenUnits = 100;
numClasses = 2;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% Option settings
options = trainingOptions('sgdm',...
    'InitialLearnRate',0.001, ...
    'Plots','training-progress',...
    'Verbose',false);

%% Prepare the training data
for ii = 1:6
    XTrain{ii} = LeftBackward1(500*(ii-1)+1:500*ii,:)';
end
for ii = 1:6
    XTrain{ii+6} = LeftBackward2(500*(ii-1)+1:500*ii,:)';
end
for ii = 1:6
    XTrain{ii+12} = RightForward1(500*(ii-1)+1:500*ii,:)';
end
for ii = 1:6
    XTrain{ii+18} = RightForward2(500*(ii-1)+1:500*ii,:)';
end
XTrain = XTrain';

YTrain = [zeros(1,12) ones(1,12)];
YTrain = YTrain';
YTrain = categorical(YTrain);

%% Train the data
net = trainNetwork(XTrain,YTrain,layers,options);

%% Prepare the test data
for ii = 1:6
    XTest{ii} = LeftBackward3(500*(ii-1)+1:500*ii,:)';
end
for ii = 1:6
    XTest{ii+6} = RightForward3(500*(ii-1)+1:500*ii,:)';
end

YTest = [zeros(1,6) ones(1,6)];
YTest = YTest';
YTest = categorical(YTest);

%% Validate the data
YPredict = classify(net, XTest);

%% Visualize the accuracy
figure;
plot(YTest, 'o');
hold on
plot(YPredict, '*');
legend('Original', 'Predicted');