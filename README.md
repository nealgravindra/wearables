# wearables



**Time-series classification**

Models that use each time-point as a feature will not be able to take advantage of the information within the sequential order of the data, leading to the common use of RNNs and CNNs to model dynamics (see `sktime-dl`)

For small data and non-DL models:
- kNN w/dynamic time warping
- TimeSeriesForest
- BOSS/cBOSS
- RISE
- Shaplet Transform Classifier
- InceptionTime
- 
