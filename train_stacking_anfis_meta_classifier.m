

     rng(20);
     load('TimefeaturesBest.mat');  % Time-based features

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
     cost1=0;
     %----------------------------------------------------------------------------
     % Define the number of folds
     k = 5;  % For example, 5-fold cross-validation
     epochNum = 60;
     % Initialize variables to store results
     metaTestAccuracies = zeros(k, 1); % Store test accuracies for each fold
     metaConfusionMatrices = cell(k, 1); % Store confusion matrices for each fold

     % Generate fold indices
     cv = cvpartition(allTimeLabels, 'KFold', k); % Ensure consistent folds across labels
     % Set up subtractive clustering options
     fisOptions = genfisOptions('SubtractiveClustering');
     fisOptions.ClusterInfluenceRange = 0.55; % Adjust for clustering
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
         %%
         % Train the meta-classifier
         % ANFIS as the meta-classifier
         fuzzyMetaOptions = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', 0.5);

         % Generate initial FIS for meta-classifier
         metaFIS = genfis(metaTrainFeatures, metaTrainLabels, fuzzyMetaOptions);

         % Train the meta-classifier using ANFIS
         epochNum = 120; % Number of training epochs
         metaFIS = anfis([metaTrainFeatures, metaTrainLabels], metaFIS, epochNum);
         % Assign weight to the first rule
         metaFIS.Rules(1).Weight = 0.096; % Example: Adjust weight to 0.8
         metaFIS.Rules(2).Weight = 0.631 ; % Example: Adjust weight to 0.8
         metaFIS.Rules(3).Weight = 0.7252; % Example: Adjust weight to 0.8

         % Predict using the trained ANFIS meta-classifier
         metaTrainPredictions = min(max(round(evalfis(metaTrainFeatures, metaFIS)), 1), 3);
         metaTestPredictions = min(max(round(evalfis(metaTestFeatures, metaFIS)), 1), 3);

         % Calculate accuracies
         metaTestAccuracies(fold) = sum(metaTestPredictions == metaTestLabels) / length(metaTestLabels) * 100;
         metaTrainAccuracies(fold) = sum(metaTrainPredictions == metaTrainLabels) / length(metaTrainLabels) * 100;
         cost1 = (1-(metaTestAccuracies/100)) + cost1;

     end

     % Display average accuracy across folds
     meanTestAccuracy = mean(metaTestAccuracies);
     fprintf('Stacking Ensemble (Meta-Classifier) - Average Test Accuracy: %.2f%%\n', meanTestAccuracy);


     % Display average training accuracy across folds
     meanTrainAccuracy = mean(metaTrainAccuracies);
     fprintf('Stacking Ensemble (Meta-Classifier) - Average Training Accuracy: %.2f%%\n', meanTrainAccuracy);
     %Return the total cost
     Cost_SMA = cost1/k;

