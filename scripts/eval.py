'''
House evaluation metrics and simulations to ensure that chosen metric 
  aligns with computational goals of the project.

Arguments:
  
'''

import torch
import sklearn.metrics as sklmetrics
import pandas as pd
import numpy as np

class random_clf():
    def __init__(self):
        self.random_clf = torch.distributions.normal.Normal(0, 1)
        self.yhat = None

    def fit(self, X, y_true):
        self.n_samples = y_true.shape[0]
        self.n_classes = 1 if len(y_true.size())==1 else y_true.shape[2]
        assert len(y_true.size()) <= 2, 'Only multi-class not multi-label or multi-output supported'

    def predict_proba(self):
        self.yhat = torch.softmax(random_clf.sample((self.n_samples, self.n_classes)), dim=-1)
        return self.yhat

class eval_evalmetric():
    '''
    Experiments to evaluate the chosen eval metric for various class imbalances.
    '''

    def __init__(self, clf=random_clf(), p_class=[[0.5], [0.4], [0.3], [0.2], [0.1], [1/100], [1/500], [1/1000]],
                 n_trials=100):
        self.p_class = p_class
        self.n_trials = n_trials
        self.clf = clf

        # logger
        self.eval = {}

    def eval_metric(self, target, output):
       '''AU-PRC for multi-class or binary classification
       '''
       precision = dict()
       recall = dict()
       auprc = dict()
       for i in range(self.n_classes):
           precision[i], recall[i] = sklmetrics.precision_recall_curve(target[:, i], output[:, i])
           auprc[i] = sklmetrics.auc(precision[i], recall[i])
           # HERE

    def get_X(self):
        return None

    def generate_y(self, p_minority=0.5, n_samples=1000, wide=False):
    '''                                                                                                 
    Arguments:                                                                                          
      p_minority (float or list): if list, probabilities of 0th, 1st, 2nd, ... and so                   
        on class. If list, must sum to 1.                                                               
      wide (bool): if True, then y_true shape: (n_sampes, n_classes), else (n_samples,)                 

    NOTE:                                                                                               
      Assumes that if p_minority type float, minority class will be assigned to the positive class      
        for binary classification task.                                                                 
    '''
    if not isinstance(p_minority, list) or len(p_minority)==1:
        p_majority = 1 - p_minority
        self.n_classes = 1
        cat_dist = torch.distributions.categorical.Categorical(probs=torch.tensor([p_majority, p_minorit\
y]))
    else:
        assert sum(p_minority)==1, 'Multi-class classification requires p sums to 1.'
        cat_dist = torch.distributions.categorical.Categorical(probs=torch.tensor(p_minority))
        self.n_classes = len(p_minority)

    y_true = cat_dist.sample((n_samples,))
    y_true_wide = torch.zeros(n_samples, n_classes)
    if wide:
        y_true_wide[torch.arange(n_samples), y_true] = 1.
        y_true = y_true_wide
    return y_true

    def trial(self, p):
        X = self.get_X()
        y = self.generate_y(p_minority=p)
        self.clf.fit(X, y)
        scores = self.clf.predict_proba()
        return self.eval_metric(y, scores) 

    def experiment(self):
        for i, class_probs in enumerate(self.p_class):
            for j, n in enumerate(range(self.n_trials)):
                result = trial(class_probs)
                label = ['p{}_'.format(p) for p in class_probs] + '_n{}'.format(n+1)
                self.evals[label] = result


            
        
        
        
    
