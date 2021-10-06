'''
Description:
  House evaluation metrics and simulations to ensure that chosen metric 
    aligns with computational goals of the project.
'''

import torch
import sklearn.metrics as sklmetrics
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

class random_clf():
    def __init__(self):
        self.random_clf = torch.distributions.normal.Normal(0, 1)
        self.yhat = None

    def fit(self, X, y_true):
        self.n_samples = y_true.shape[0]
        self.n_classes = 1 if len(y_true.size())==1 else y_true.shape[1]
        assert len(y_true.size()) <= 2, 'Only multi-class not multi-label or multi-output supported'

    def predict_proba(self):
        if self.n_classes > 1:
            self.yhat = torch.softmax(self.random_clf.sample((self.n_samples, self.n_classes)), dim=-1)
        else:
            self.yhat = torch.sigmoid(self.random_clf.sample((self.n_samples, self.n_classes)))
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

    def eval_metric(self, target, output, mode='AP', nan20=False):
        '''AU-PRC for multi-class or binary classification
        '''
        precision = dict()
        recall = dict()
        metric = dict()
        if self.n_classes==1:
            target = target.unsqueeze(1)
        for i in range(self.n_classes):
            precision[i], recall[i], _ = sklmetrics.precision_recall_curve(
                    target[:, i], output[:, i])
            if 'AP' in mode:
                metric[i] = sklmetrics.average_precision_score(target[:, i], output[:, i])
                metric[i] = 1. if np.isnan(metric[i]) else metric[i] # https://github.com/scikit-learn/scikit-learn/issues/8245 
            elif 'AU-PRC' in mode:
                metric[i] = sklmetrics.auc(recall[i], precision[i]) 
        self.metric = metric # store metric
        if nan20:
            # replace nan with 0 
            metric = {k:v if not np.isnan(v) else 0. for k,v in metric.items()}
        if 'weighted' in mode and self.n_classes > 1: 
            metric_macroave = np.nansum([self.p_minority[i]*metric[i] for i in range(self.n_classes)])
        elif 'inverse' in mode and self.n_classes > 1:
            metric_macroave = np.nansum([metric[i]*(1./self.p_minority[i]) for i in range(self.n_classes)])
        else:
            metric_macroave = np.nanmean([v for v in metric.values()]) # could be artificially high
        return metric_macroave

    def get_X(self):
        return None

    def generate_y(self, p_minority=0.5, n_samples=1000, wide=True):
        '''Assign positive class to minority class. 

        Arguments:
          p_minority (float or list): if list, probabilities of 0th, 1st, 2nd, ... and so
            on class. If list, must sum to 1.
          wide (bool): if True, then y_true shape: (n_sampes, n_classes), else (n_samples,)

        NOTE:
          Assumes that if p_minority type float, minority class will be assigned to the 
            positive class for binary classification task.
        '''
        if not isinstance(p_minority, list) or len(p_minority)==1:
            if isinstance(p_minority, list):
                p_minority = p_minority[0]
            p_majority = 1 - p_minority
            self.n_classes = 1
            cat_dist = torch.distributions.categorical.Categorical(probs=torch.tensor([p_majority, p_minority]))
        else:
            assert sum(p_minority)==1, 'Multi-class classification requires p sums to 1.'
            cat_dist = torch.distributions.categorical.Categorical(probs=torch.tensor(p_minority))
            self.n_classes = len(p_minority)
        self.p_minority = p_minority
        y_true = cat_dist.sample((n_samples,))
        if wide and self.n_classes > 1:
            y_true_wide = torch.zeros(n_samples, self.n_classes, dtype=int)
            y_true_wide[torch.arange(n_samples), y_true] = 1
            y_true = y_true_wide
        return y_true

    def trial(self, p, mode, nan20):
        X = self.get_X()
        y = self.generate_y(p_minority=p)
        self.clf.fit(X, y)
        scores = self.clf.predict_proba()
        return self.eval_metric(y, scores, mode=mode, nan20=nan20) 

    def experiment(self, mode='AP', nan20=False, verbose=True):
        if verbose:
            tic = time.time()
        for i, class_probs in enumerate(self.p_class):
            for j, n in enumerate(range(self.n_trials)):
                result = self.trial(class_probs, mode=mode, nan20=nan20)
                label = ''.join('p{}_'.format(p) for p in class_probs) + 'n{}'.format(n+1)
                self.eval[label] = result
                if verbose:
                    print('  completed {}-th class probability, {}-th trial'.format(i+1, j+1))
        if verbose:
            print('\n... {} exps with n={} trials per exp completed in {:.1f}-s'.format(
                len(self.p_class), self.n_trials, time.time()-tic))
        self.summary = {k_:[] for k_ in np.unique([k.split('_n')[0] for k in self.eval.keys()])}
        for k, v in self.eval.items():
            self.summary[k.split('_n')[0]].append(v)
        res = pd.melt(pd.DataFrame(self.summary), var_name='P_class', value_name=mode)
        return res

def auprc_adjusted(n_trials=100):
    '''Class imbalance adjusted AU-PRC for binary or multi-class classification

    Arguments:
      outputs
      targets

    NOTE:
      - simulation assumes random classifier produced by Gaussian distributed
        probabilities and draws from training set to select minority as positive class
    '''

    # simulate random model
    
    return None

if __name__ == '__main__':
    mode = 'weighted_AU-PRC'
    nan20 = False
    p_class=[
            [0.5, 0.4, 0.1],
            [0.5, 0.49, 0.01],
            [0.5, 0.499, 0.001],
            [0.8, 0.1, 0.1],
            [0.9, 0.09, 0.01],
            [0.9, 0.099, 0.001]
            ]
    # p_class = [[0.5], [0.6], [0.7], [0.8], [0.9], [99/100], [999/1000]]
    res_multi = eval_evalmetric(p_class=p_class).experiment(verbose=False, mode=mode, nan20=nan20)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    p = sns.boxplot(x='P_class', y=mode, data=res_multi, ax=ax)
    p.set_xticklabels(p.get_xticklabels(), rotation=45)
    fig.tight_layout()
    plt.show()
