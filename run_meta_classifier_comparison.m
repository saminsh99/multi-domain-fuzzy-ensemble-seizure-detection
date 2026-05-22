% Load each feature dataset
rng(20);
load('TimefeaturesBest.mat');  % Time-based features
dataTime = [featuresA, featuresB, featuresC, featuresD, featuresE];

dataTime1 = featuresE; % Seizure samples
dataTime2 = featuresC; % Epileptic samples
dataTime3 = featuresD; % Epileptic samples
dataTime4 = featuresA; % Healthy samples
dataTime5 = featuresB; % Healthy samples
%%
load('Freqfeatures2.mat');  % Frequency-based features
dataFreq = [featuresA, featuresB, featuresC, featuresD, featuresE];

dataFreq1 = featuresE; % Seizure samples
dataFreq2 = featuresC; % Epileptic samples
dataFreq3 = featuresD; % Epileptic samples
dataFreq4 = featuresA; % Healthy samples
dataFreq5 = featuresB; % Healthy samples
%%
load('TimeFreqfeatures1.mat');  % Wavelet-based features
dataWavelet = [featuresA, featuresB, featuresC, featuresD, featuresE];

dataWavelet1 = featuresE; % Seizure samples
dataWavelet2 = featuresC; % Epileptic samples
dataWavelet3 = featuresD; % Epileptic samples
dataWavelet4 = featuresA; % Healthy samples
dataWavelet5 = featuresB; % Healthy samples
%%
% Preprocess each dataset (normalization, labels, split into train/test)
allTimeData = [dataTime1, dataTime2, dataTime3, dataTime4, dataTime5];
allTimeLabels = [ones(1, size(dataTime1, 2)), 2 * ones(1, size(dataTime2, 2)), 2 * ones(1, size(dataTime3, 2)),...
    3 * ones(1, size(dataTime4, 2)), 3 * ones(1, size(dataTime5, 2))];

allFreqData = [dataFreq1, dataFreq2, dataFreq3, dataFreq4, dataFreq5];
allFreqLabels = [ones(1, size(dataFreq1, 2)), 2 * ones(1, size(dataFreq2, 2)), 2 * ones(1, size(dataFreq3, 2)),...
    3 * ones(1, size(dataFreq4, 2)), 3 * ones(1, size(dataFreq5, 2))];

allWaveletData = [dataWavelet1, dataWavelet2, dataWavelet3, dataWavelet4, dataWavelet5];
allWaveletLabels = [ones(1, size(dataWavelet1, 2)), 2 * ones(1, size(dataWavelet2, 2)), 2 * ones(1, size(dataWavelet3, 2)),...
    3 * ones(1, size(dataWavelet4, 2)), 3 * ones(1, size(dataWavelet5, 2))];


% Normalize features to [0, 1]
allTimeData = (allTimeData - min(allTimeData, [], 2)) ./ (max(allTimeData, [], 2) - min(allTimeData, [], 2));


allFreqData = (allFreqData - min(allFreqData, [], 2)) ./ (max(allFreqData, [], 2) - min(allFreqData, [], 2));


allWaveletData = (allWaveletData - min(allWaveletData, [], 2)) ./ (max(allWaveletData, [], 2) - min(allWaveletData, [], 2));


% Transpose allData and allLabels for easier handling
allTimeData = allTimeData';
allTimeLabels = allTimeLabels';

allFreqData = allFreqData';
allFreqLabels = allFreqLabels';

allWaveletData = allWaveletData';
allWaveletLabels = allWaveletLabels';

%----------------------------------------------------------------------------
% Define the number of folds
k = 5;  % For example, 5-fold cross-validation
fisOptions = genfisOptions('SubtractiveClustering');
fisOptions.ClusterInfluenceRange = 0.55; % Adjust range for clustering
epochNum=150;
% Generate FIS using subtractive clustering
%fis = genfis(anfisTrainData(:, 1:end-1), anfisTrainData(:, end), fisOptions);

% Initialize variables to store results
metaTestAccuracies = zeros(k, 1); % Store test accuracies for each fold
metaConfusionMatrices = cell(k, 1); % Store confusion matrices for each fold

% Generate fold indices
cv = cvpartition(allTimeLabels, 'KFold', k); % Ensure consistent folds across labels

% Perform k-fold cross-validation
for fold = 1:k
    fprintf('Fold %d/%d\n', fold, k);
    
    % Training and testing indices for this fold
    trainIndices = training(cv, fold);
    testIndices = test(cv, fold);

    % Train data for each feature type
    trainFeaturesTime = allTimeData(trainIndices, :);
    trainLabelsTime = allTimeLabels(trainIndices);
    testFeaturesTime = allTimeData(testIndices, :);
    testLabelsTime = allTimeLabels(testIndices);

    trainFeaturesFreq = allFreqData(trainIndices, :);
    trainLabelsFreq = allFreqLabels(trainIndices);
    testFeaturesFreq = allFreqData(testIndices, :);
    testLabelsFreq = allFreqLabels(testIndices);

    trainFeaturesWavelet = allWaveletData(trainIndices, :);
    trainLabelsWavelet = allWaveletLabels(trainIndices);
    testFeaturesWavelet = allWaveletData(testIndices, :);
    testLabelsWavelet = allWaveletLabels(testIndices);

    % Train ANFIS models for each feature set
    anfisModelTime = anfis([trainFeaturesTime, trainLabelsTime], ...
                           genfis(trainFeaturesTime(:, 1:35), trainLabelsTime, fisOptions), epochNum);
    anfisModelFreq = anfis([trainFeaturesFreq, trainLabelsFreq], ...
                           genfis(trainFeaturesFreq(:, 1:35), trainLabelsFreq, fisOptions), epochNum);
    anfisModelWavelet = anfis([trainFeaturesWavelet, trainLabelsWavelet], ...
                              genfis(trainFeaturesWavelet(:, 1:42), trainLabelsWavelet, fisOptions), epochNum);

    % Generate predictions for training data
    trainPredTime = min(max(round(evalfis(trainFeaturesTime, anfisModelTime)), 1), 3);
    trainPredFreq = min(max(round(evalfis(trainFeaturesFreq, anfisModelFreq)), 1), 3);
    trainPredWavelet = min(max(round(evalfis(trainFeaturesWavelet, anfisModelWavelet)), 1), 3);

    % Generate predictions for testing data
    testPredTime = min(max(round(evalfis(testFeaturesTime, anfisModelTime)), 1), 3);
    testPredFreq = min(max(round(evalfis(testFeaturesFreq, anfisModelFreq)), 1), 3);
    testPredWavelet = min(max(round(evalfis(testFeaturesWavelet, anfisModelWavelet)), 1), 3);

    % Meta-Classifier training data
    metaTrainFeatures = [trainPredTime, trainPredFreq, trainPredWavelet];
    metaTrainLabels = trainLabelsTime;

    metaTestFeatures = [testPredTime, testPredFreq, testPredWavelet];
    metaTestLabels = testLabelsTime;
    %% tree bagg
    % Train the meta-classifier
    [metaClassifierbagg] = fitcensemble(metaTrainFeatures, metaTrainLabels, 'Method', 'Bag');
    metaTestPredictionsbagg = predict(metaClassifierbagg, metaTestFeatures);
    metaTrainPredictionsbagg = predict(metaClassifierbagg, metaTrainFeatures);
    metaTestAccuraciesbagg(fold) = sum(metaTestPredictionsbagg == metaTestLabels) / length(metaTestLabels) * 100;
    metaTrainAccuraciesbagg(fold) = sum(metaTrainPredictionsbagg == metaTrainLabels) / length(metaTrainLabels) * 100;

    %% knn
    % Train the meta-classifier
    [metaClassifierknn] = fitcknn(metaTrainFeatures, metaTrainLabels, 'NumNeighbors', 5);
    metaTestPredictionsknn = predict(metaClassifierknn, metaTestFeatures);
    metaTrainPredictionsknn = predict(metaClassifierknn, metaTrainFeatures);
    metaTestAccuraciesknn(fold) = sum(metaTestPredictionsknn == metaTestLabels) / length(metaTestLabels) * 100;
    metaTrainAccuraciesknn(fold) = sum(metaTrainPredictionsknn == metaTrainLabels) / length(metaTrainLabels) * 100;

    %% ann
    % Train the meta-classifier
    metaClassifierann = patternnet(10); % 10 hidden neurons
    [metaClassifierann] = train(metaClassifierann, metaTrainFeatures', dummyvar(metaTrainLabels)'); % Train with one-hot labels
    metaTestPredictionsann = vec2ind(metaClassifierann(metaTestFeatures'))'; % Convert one-hot output to class labels
    metaTrainPredictionsann = vec2ind(metaClassifierann(metaTrainFeatures'))'; % Convert one-hot output to class labels
    metaTestAccuraciesann(fold) = sum(metaTestPredictionsann == metaTestLabels) / length(metaTestLabels) * 100;
    metaTrainAccuraciesann(fold) = sum(metaTrainPredictionsann == metaTrainLabels) / length(metaTrainLabels) * 100;

    %% svm
    % Train the meta-classifier
    [metaClassifiersvm] = fitcecoc(metaTrainFeatures, trainLabelsTime); % Multiclass SVM

    metaTestPredictionssvm = predict(metaClassifiersvm, metaTestFeatures);
    metaTrainPredictionssvm = predict(metaClassifiersvm, metaTrainFeatures);
    metaTestAccuraciessvm(fold) = sum(metaTestPredictionssvm == metaTestLabels) / length(metaTestLabels) * 100;
    metaTrainAccuraciessvm(fold) = sum(metaTrainPredictionssvm == metaTrainLabels) / length(metaTrainLabels) * 100;

    %% ANFIS
    % Train the meta-classifier
    
    fuzzyMetaOptions = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', 0.7);
    metaFIS = genfis(metaTrainFeatures, metaTrainLabels, fuzzyMetaOptions);
    epochNum = 150;
    [metaClassifierfis,trainError] = anfis([metaTrainFeatures, metaTrainLabels], metaFIS, epochNum);
    % Predict using the trained ANFIS meta-classifier
    metaTrainPredictionsfis = min(max(round(evalfis(metaTrainFeatures, metaClassifierfis)), 1), 3);
    metaTestPredictionsfis = min(max(round(evalfis(metaTestFeatures, metaClassifierfis)), 1), 3);

 
    metaTestAccuraciesfis(fold) = sum(metaTestPredictionsfis == metaTestLabels) / length(metaTestLabels) * 100;
    metaTrainAccuraciesfis(fold) = sum(metaTrainPredictionsfis == metaTrainLabels) / length(metaTrainLabels) * 100;
    
  
end

% Display average accuracy across folds
meanTestAccuracybagg = mean(metaTestAccuraciesbagg);
fprintf('Stacking Ensemble (Meta-Classifier= tree bagging) - Average Test Accuracy: %.2f%%\n', meanTestAccuracybagg);

meanTestAccuracyknn = mean(metaTestAccuraciesknn);
fprintf('Stacking Ensemble (Meta-Classifier= knn) - Average Test Accuracy: %.2f%%\n', meanTestAccuracyknn);

meanTestAccuracysvm = mean(metaTestAccuraciessvm);
fprintf('Stacking Ensemble (Meta-Classifier = svm) - Average Test Accuracy: %.2f%%\n', meanTestAccuracysvm);

meanTestAccuracyann = mean(metaTestAccuraciesann);
fprintf('Stacking Ensemble (Meta-Classifier = ann) - Average Test Accuracy: %.2f%%\n', meanTestAccuracyann);

meanTestAccuracyfis = mean(metaTestAccuraciesfis);
fprintf('Stacking Ensemble (Meta-Classifier = anfis) - Average Test Accuracy: %.2f%%\n', meanTestAccuracyfis);






