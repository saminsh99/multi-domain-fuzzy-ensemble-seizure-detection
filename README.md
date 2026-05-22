# multi-domain-fuzzy-ensemble-seizure-detection
A Multi-Domain Fuzzy Ensemble Approach for Epileptic Seizure Detection

This repository contains the MATLAB implementation associated with the paper:

**A Multi-Domain Fuzzy Ensemble Approach for Epileptic Seizure Detection**  
Published in: *2025 33rd International Conference on Electrical Engineering (ICEE)*  
DOI: [10.1109/ICEE67339.2025.11213726](https://doi.org/10.1109/ICEE67339.2025.11213726)

## Overview

This project presents an interpretable fuzzy ensemble framework for epileptic seizure detection using EEG signals. The method extracts features from multiple signal domains, trains separate ANFIS models for each domain, and combines their outputs using a meta-classifier.

The main objective is to improve classification performance while preserving interpretability through fuzzy rule-based decision-making.

## Method Summary

The proposed pipeline includes:

1. Loading EEG segments from the Bonn EEG dataset
2. Signal preprocessing
3. Feature extraction from three domains:
   - Time domain
   - Frequency domain
   - Wavelet domain
4. Training separate ANFIS models for each feature domain
5. Combining domain-specific predictions using a stacking-based meta-classifier
6. Evaluating the final model using classification metrics and cross-validation

## Feature Domains

### Time-Domain Features
The time-domain feature set includes statistical and signal-shape features such as mean, variance, skewness, kurtosis, line length, nonlinear energy, and derivative-based features.

### Frequency-Domain Features
Frequency-domain features are extracted using spectral analysis methods such as FFT and power-related measures.

### Wavelet-Domain Features
Wavelet-domain features are extracted using the discrete wavelet transform with a Daubechies wavelet. These features capture time-frequency characteristics of the signal.

## Model

The classification framework is based on Adaptive Neuro-Fuzzy Inference System (ANFIS) models. Three ANFIS models are trained separately using time-domain, frequency-domain, and wavelet-domain features. Their predictions are then combined using a meta-classifier.

Different meta-classifiers were evaluated in the original study, including KNN, SVM, TreeBagger, ANN, and ANFIS.

Dataset

This study uses the Bonn EEG dataset for epileptic seizure detection.

The dataset is not included in this repository. Users should download the dataset from its official source and place the files in the following structure:
( converted the dataset to .mat format )
# data/
├── A/
├── B/
├── C/
├── D/
└── E/
