clc
clear
close all
%% load data
load A.mat
load B.mat
load C.mat
load D.mat
load E.mat

fs=173.61;% sampling frequency
%% wavelet parameters
wname= 'db6';
level=6;

Nf=(level+1)*9;% number of features(coeff*features=49)
Nt= size(A,2); % number of trials
featuresA=zeros(Nf,Nt);
featuresB=zeros(Nf,Nt);
featuresC=zeros(Nf,Nt);
featuresD=zeros(Nf,Nt);
featuresE=zeros(Nf,Nt);

%% design stop filer
%[fl fh]: A vector containing the low and high cutoff frequencies of the filter.
%fs: The sampling frequency.
%type: The type of filter to design (e.g., lowpass, highpass, bandpass, etc.).
fl=49.9;
fh=50.1;
order= 3; %use information from the current and three previous samples to calculate the filtered output
type= 'stop'; %a stop filter eliminates certain frequencies within a specified band while allowing all other frequencies to pass through unaffected.
[b,a]= butter(order,[fl fh]/ (fs/2), type);
for i=1:Nt
    sigA= A(:,i);
    sigB= B(:,i);
    sigC= C(:,i);
    sigD= D(:,i);
    sigE= E(:,i);
    %% step 1: denoising
    % apply notch (stop) filter
    sigA= filtfilt(b,a,sigA);
    sigB= filtfilt(b,a,sigB);
    sigC= filtfilt(b,a,sigC);
    sigD= filtfilt(b,a,sigD);
    sigE= filtfilt(b,a,sigE);
    %% step 2: feature extraction in wavelet domain
    % step 2-5: wavelet transform
    [Ca,L]= wavedec(sigA,level,wname);
    [Cb,~]= wavedec(sigB,level,wname);
    [Cc,~]= wavedec(sigC,level,wname);
    [Cd,~]= wavedec(sigD,level,wname);
    [Ce,~]= wavedec(sigE,level,wname);
    
    L=[0;L];
    %%
    %take all approximate and detailes step by step
    %numel= number of elements
    %it extracts the corresponding coefficients Ca_j, Cb_j, Cc_j, Cd_j, and Ce_j from the wavelet decomposition results for each signal.
    for j=2:numel(L)-1
        indx= sum(L(1:j-1))+1:sum(L(1:j));
        Ca_j= Ca(indx);
        Cb_j= Cb(indx);
        Cc_j= Cc(indx);
        Cd_j= Cd(indx);
        Ce_j= Ce(indx);

        %The extracted coefficients Ca_j, Cb_j, Cc_j, Cd_j, and Ce_j are passed to the function myfeatureExtraction, which computes features based on these coefficients.
        %The resulting features are stored in temporary variables tpA, tpB, tpC, tpD, and tpE.
        tpA(:,j-1) = myfeatureExtraction(Ca_j);
        tpB(:,j-1) = myfeatureExtraction(Cb_j);
        tpC(:,j-1) = myfeatureExtraction(Cc_j);
        tpD(:,j-1) = myfeatureExtraction(Cd_j);
        tpE(:,j-1) = myfeatureExtraction(Ce_j);

    end  
    %After the loop completes, the features extracted for each signal in the current trial are stored in the respective columns of the feature matrices featuresA, featuresB, featuresC, featuresD, and featuresE.
    featuresA(:,i) = tpA(:);
    featuresB(:,i) = tpB(:);
    featuresC(:,i) = tpC(:);
    featuresD(:,i) = tpD(:);
    featuresE(:,i) = tpE(:);
    disp(['iteration: ',num2str(i)])
end
%After processing all trials, the extracted features for each dataset are saved in the file 'TimeFreqfeatures1.mat'.
save wavelet9features featuresA featuresB featuresC featuresD featuresE



