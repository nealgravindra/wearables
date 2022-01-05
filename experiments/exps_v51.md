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
