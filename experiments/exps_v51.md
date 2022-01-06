# Experiments

## Does learning rate scheduler improve performance?

Disappointing results, possibly because of size of learning rate. The training is also slow.

## Does the nb_layers influence performance much?

Can we reduce the model parameters, using just MSE and L2 reg?

## Runs

### Parameters to modify

- exp
- trial
- cuda_nb
- nb_layers
- scheduler (true/false)
- criterion (mse for most, but one with msel1)
    - *NOTE*: the lambdas may have changed but this is a separate optimization choice
    
## Do we have a bias or variance problem?

### Comparing error vs. split size (train v. test)

We can run this with simpler models too.

# Outstanding ToDos

- [ ] 1. Redo error analysis with a selected model and use the embeddings to perform other metadata tasks 
- [ ] 2. leaner data loader (store only ids used per model, not full dataset)
- [ ] 3. fixed vgg
- [ ] 4. transfer learning methods with NHANES data
- [ ] 5. fancier models (like ConViT)
- [ ] 6. try contrastive loss in minibatch for gt/lt, penelizing when there are paired measurements in the mini-batch and model assigns higher value to later one (will benefit to have batch size of 64 rather than 32)

# Ideas

## performance

A few ideas to improve the performance are:
- add metadata variables as features to the network (see tft as to how to add static feats)
- interpolate to reduce features to model
- transformer-based (try tft)
- improve loss function (patient greater than or equal to)
- try to predict a handful of interesting md vars with main model, forget about GA and do the class imbalance or multi-task + class-imbalance correction from ts

## implementation

### PyTorch Forecasting
 
This is a tool, perhaps similar to PyTorchGeometric, which is made to handle ts data. It might be easier to work with but the data has to be converted into a long pd.DataFrame, where the 'group' column indicates the ts, and other columns can be features. In theory, this allows for separate encoding of static and dynamic features, as well as categorical and continuous values in either static and dynamic types. This is [their description](https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/building.html#Passing-data-to-a-model):
>The data has to be in a specific format to be used by the TimeSeriesDataSet. It should be in a pandas DataFrame and have a categorical column to identify each series and a integer column to specify the time of the record.

- port to pytorch lightning? Allows for multi-GPU training (potentially) and better logging practices.
- 

# Figures

Overall, we call our new algorithm BLAH, which builds on an architecture developed for time-series classification, InceptionTime, which we adapt for regression, unsupervised clustering using graph-based methods, and recent feature attribution methods from the field of explainable AI. 

## 1: model comparisons and training curves compared to random, supplement with training set size?

We need to prove that we can at least learn something from the data, e.g., that we're better than other methods, and that we actually do learn something (train/test loss curves) but that we still have a bias or variance problem that will have to be addressed in the future. Nonetheless, is this prediction useful for other things?

## 2: error analysis showing that ptb is most predictable and error groups are significantly indicative of ptb 

This is useful because if we know the GA and what the model thinks your GA is, we can indicate whether you might be in a ptb phenotypic group

# 3: is model learning (its embedding) learning something useful -- unsupervised clustering of model embeddings relative to DTW AND model embedding predictions per metadata var

# 4: since model is learning something useful, can we interpret why it is making these decisions, e.g., is it related to activity and sleep, i.e., feat attributions stratified by time of day, week, sleep v. week AND ptb
