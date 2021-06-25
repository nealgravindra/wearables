# wearables

**Keywords**: mobility data, gestational age

## Background

### Biological

### Methodological

**Time-series classification**

Models that use each time-point as a feature cannot take advantage of the information within the sequential order of the data (e.g., KRR on aligned & fixed-width sequences). To incorporate this information, RNNs and CNNs (e.g., InceptionTime) are commonly used model dynamics (see `sktime-dl`). 

For small data and non-DL models, implemented in `sktime`:
- kNN w/dynamic time warping
  -  use DTW as a distance metric for a k-NN classifier
  -  *advantage*: good benchmark because of its simplicity
  -  *disadvantage*: performs poorly if noise in series is larger than subtle differences in shape
- TimeSeriesForest
  -  RF for time series by splitting series into random sub-sets with random start/length/stop points, extracting primitive values from each interval, and training a RF on those extracted features per series
  -  *advantage*: can be interpreted
- BOSS/cBOSS
  - dictionary-based that uses a sliding window to transform series into a "word" of specified length and of specified number of letters 
  - apply Symbolic Fourier Approximation (SFA) transformation by calculating FT on each window, discretizing first `l` Fourier terms and concatenating to form a "word" using Multiple Coefficient Binning (MCB), a supervised method that bins continuous time-series into a sequence of letters. As the window slides, a count of each word's frequency is recorded in a dictionary, then converted into a histogram. Then, any classifier can be trained on these word histograms and the words can be interpretted (BOSS usess nearest-neighbor classifier and gridsearch to select length of words, number of letters, FT normalization; cBOSS is a contractable version, randomly sampling from parameter space w/o replacement and retaining a fixed-number of classifiers, since BOSS gridsearch has large memory and time requirements w/o much loss in performance)
  - *disadvantage*: large time and memory requirements and grid-search procedure; also, classifiers may be too simplistic (kNN)
  - *intuition*: dictionary-based classifiers perform well when you can discriminate using the frequency of a particular pattern
- RISE
  - single random interval from time-series, and instead of extracted primitive values, and relies on concatenated spectral features of the interval for RF classifier, e.g., fitted auto-regressive coefs., est. autocorr coefs., and power spectrum coefs. (i.e., *series-to-series feature extraction transformers*)
- Shaplet Transform Classifier
  - enumerate intervals, to extract sub-shapes of time-series, which are useful to detect "phase-independent localized similarity between series". After identifying the top-k shapelets (usually evaluated by information gain, and strongest non-overlapping shapelets are retained) to become k features, any classifier can be trained (in the past, weighted ensemble classifier, and Rotation Forest, which is a tree-based ensemble on PCA of the input features, have been used... for continuous features, rotation forest works well but is not available in python)
  - *advantage*: shapelet features can be used for interpretability... presence of shapelets may indicate something discriminative 
  - *disadvantage*: enumeration of all intervals can be costly, so `sktime` randomly searches for shapelets, potentially missing something important
  - *intuition*: shapelet-based classifiers perform well when the "best" feature is the presence or absence of a phase-independent pattern in a series
- HIVE-COTE or ROCKET
  - the HIVE-COTE ensemble classifier can combine all of the above (Hierarchical Vote Collective of Transformation-based Ensembles), e.g., per time-series, feed to Shaplet Transform, TimeSeriesForest, RISE, and cBOSS, then take a weighted average of predictions with a control unit that assigns weights to classifiers by their quality. 
  - ROCKET is a simple linear classifier based on random convolutional kernels such as random length, weight, dilation, padding, etc.

Feature extraction of time series can be *global* or *local* (sliding windows, or bins), transforming the time series into *primitive* values (mean, sd, etc.) or other *series* (FT, series of auto-regression coefs.).

REF: Alexandra Amidon blog on `sktime`, see [here](https://towardsdatascience.com/a-brief-introduction-to-time-series-classification-algorithms-7b4284d31b97). `sktime` can be found [here](https://github.com/alan-turing-institute/sktime). Loning et al. sktime, *NeurIPS*, 2019

**NOTE**: the primary goal with wearables is to convert these models for use in *time-series regression* tasks, where a time-series is used to predict an output value in order to do substite testing in healthcare (e.g., AppleWatch using activity monitoring to estimate a patients six-minute walk test value). However, the above are useful in predicting some other interesting secondary-outcomes of interest in order to gauge how much *useful* information the time-series contains.

**Time-series regression**

## Data

### Mobility or activity data
- https://www.kaggle.com/shambhavimalik/activity-data
- 


## Future directions

### Development roadmap

1. Incorporate more DL models for ECG based data, use fewer and fewer leads (starting from 12-lead ECG data)
2. 

## References

1. On the broad utility of activity monitoring: https://doi.org/10.1126/science.abc5096 and how to model with atypical architectures (i.e., GNNs): https://arxiv.org/pdf/2007.03113.pdf; using Google's Community Mobility Reports (https://www.google.com/covid19/mobility/) and *The Times* COVID-19 database (https://github.com/nytimes/covid-19-data)
2. 
