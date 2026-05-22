rng(20); % Set seed for reproducibility
% Load wavelet feature data
load('TimeFreqfeatures1.mat');

data1 = featuresE; % Seizure samples
data2 = featuresC; % Epileptic samples
data3 = featuresD; % Epileptic samples
data4 = featuresA; % Healthy samples
data5 = featuresB; % Healthy samples

% Combine data and create labels
allData = [data1, data2, data3, data4, data5];
allLabels = [ones(1, size(data1, 2)), 2 * ones(1, size(data2, 2)), 2 * ones(1, size(data3, 2)), ...
             3 * ones(1, size(data4, 2)), 3 * ones(1, size(data5, 2))];

% Normalize features to [0, 1]
allData = (allData - min(allData, [], 2)) ./ (max(allData, [], 2) - min(allData, [], 2));

% Transpose data and labels for easier handling
allData = allData';
allLabels = allLabels';

% Set up k-fold cross-validation
k = 5; % Number of folds
cv = cvpartition(size(allData, 1), 'KFold', k);

% Initialize metrics
overallTrainAccuracy = zeros(1, k);
overallTestAccuracy = zeros(1, k);

% Initialize metrics
confusionMatrices = zeros(3, 3, k); % For storing confusion matrices for each fold
% Loop through each fold
for fold = 1:k
    fprintf('Processing Fold %d/%d...\n', fold, k);
    
    % Split training and test sets for this fold
    trainIndices = training(cv, fold);
    testIndices = test(cv, fold);
    
    trainFeatures = allData(trainIndices, :);
    trainLabels = allLabels(trainIndices);
    testFeatures = allData(testIndices, :);
    testLabels = allLabels(testIndices);
    
    % Prepare ANFIS training data
    anfisTrainData = [trainFeatures, trainLabels];
    
    % Set up subtractive clustering options
    fisOptions = genfisOptions('SubtractiveClustering');
    fisOptions.ClusterInfluenceRange = 0.544; % Adjust for clustering
    
    % Generate FIS using subtractive clustering
    fis = genfis(anfisTrainData(:, 1:end-1), anfisTrainData(:, end), fisOptions);
    
    % Train the ANFIS model
    epochNum = 150;
    [anfisModel, trainError] = anfis(anfisTrainData, fis, epochNum);
    
    % Evaluate on training data
    trainPredictions = evalfis(trainFeatures, anfisModel);
    predictedTrainLabels = min(max(round(trainPredictions), 1), 3); % Clamp predictions to [1, 3]
    trainAccuracy = sum(predictedTrainLabels == trainLabels) / length(trainLabels) * 100;
    overallTrainAccuracy(fold) = trainAccuracy;
    
    % Evaluate on test data
    testPredictions = evalfis(testFeatures, anfisModel);
    predictedTestLabels = min(max(round(testPredictions), 1), 3); % Clamp predictions to [1, 3]
    testAccuracy = sum(predictedTestLabels == testLabels) / length(testLabels) * 100;
    overallTestAccuracy(fold) = testAccuracy;

    % Compute confusion matrix for this fold
    confMat = confusionmat(testLabels, predictedTestLabels, 'Order', [1, 2, 3]);
    confusionMatrices(:, :, fold) = confMat;
end

% Calculate and display crisp average accuracies
meanTrainAccuracy = mean(overallTrainAccuracy);
meanTestAccuracy = mean(overallTestAccuracy);

% Calculate average confusion matrix across folds
meanConfusionMatrix = mean(confusionMatrices, 3);

% Calculate sensitivity and specificity
sensitivity = diag(meanConfusionMatrix) ./ sum(meanConfusionMatrix, 2); % TP / (TP + FN)
specificity = zeros(3, 1);
for classIdx = 1:3
    trueNegatives = sum(meanConfusionMatrix(:)) - sum(meanConfusionMatrix(classIdx, :)) - sum(meanConfusionMatrix(:, classIdx)) + meanConfusionMatrix(classIdx, classIdx);
    falsePositives = sum(meanConfusionMatrix(:, classIdx)) - meanConfusionMatrix(classIdx, classIdx);
    specificity(classIdx) = trueNegatives / (trueNegatives + falsePositives);
end

% Display results
fprintf('\nMean Confusion Matrix:\n');
disp(meanConfusionMatrix);
fprintf('\nSensitivity for each class:\n');
disp(sensitivity);
fprintf('\nSpecificity for each class:\n');
disp(specificity);
%}
%%
fprintf('\nCrisp Average Training Accuracy: %.2f%%\n', meanTrainAccuracy);
fprintf('Crisp Average Test Accuracy: %.2f%%\n', meanTestAccuracy);
%%
showrule(fis);

ruleedit(fis)
 % Check the number of membership functions for each input
numInputs = numel(fis.Inputs); % Number of input variables
for i = 1:numInputs
    numMFs = numel(fis.Inputs(i).MembershipFunctions); % Number of MFs for the i-th input
    fprintf('Input %d has %d membership functions.\n', i, numMFs);
end

%%
% Calculate the average confusion matrix
meanConfusionMatrix = mean(confusionMatrices, 3);

% Plot the confusion matrix
figure;
imagesc(meanConfusionMatrix); % Create a heatmap
colormap('sky'); % Choose a colormap (e.g., 'jet', 'hot', 'cool')
colorbar; % Add a colorbar to show the scale

% Add text annotations
numClasses = size(meanConfusionMatrix, 1);
for i = 1:numClasses
    for j = 1:numClasses
        text(j, i, sprintf('%d', round(meanConfusionMatrix(i, j))), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontSize', 12, ...
            'Color', 'black'); % Add numerical values to the plot
    end
end

% Set axes labels
xticks(1:numClasses);
yticks(1:numClasses);
xticklabels({'Seizure', 'Epileptic', 'Healthy'}); % Adjust labels as per your classes
yticklabels({'Seizure', 'Epileptic', 'Healthy'}); % Adjust labels as per your classes

% Set axes titles
xlabel('Predicted Class');
ylabel('True Class');

% Add a title
title('Confusion Matrix');

