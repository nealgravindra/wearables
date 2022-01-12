'''
Description:
  House evaluation metrics and simulations to ensure that chosen metric 
    aligns with computational goals of the project.
'''
import os
import sys
import torch
import pickle
import sklearn.metrics as sklmetrics
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# metrics 
#   for reg: (MAE, MAPE [mean absolute percentage error], Spearman's rho, P_spearman) 
#   for clf: (adjusted AU-PRC, balanced acc adj)
def auprc(output, target, nan20=False):
    '''un-balanced macro-average
    '''
    precision = dict()
    recall = dict()
    metric = dict()
    for i in range(output.shape[1]):
        precision[i], recall[i] = sklmetrics.precision_recall_curve(target[:, i], output[:, i])
        metric[i] = sklmetrics.auc(recall[i], precision[i])
    if nan20:
        metric = {k:v if not np.isnan(v) else 0. for k,v in metric.items()}
    return np.nanmean([v for v in metric.values()])

def eval_output(output, target, tasktype='regression', n_trials=10, nan20=False):
    '''Evaluate the model output (logits) versus ground-truth.
    
    Arguments:
      output (torch.tensor OR np.ndarray): y_hat
      target (torch.tensor OR np.ndarray): y_true
    '''
    
    
    if tasktype == 'regression':
        # Spearman's Rho
        if not isinstance(output, np.ndarray):
            rho, p = spearmanr(output.numpy(), target.numpy())
            mae = (output - target).abs().mean().item()
            mape = ((output - target)/target).abs().mean().item()
        else: 
            rho, p = spearmanr(output, target)
            mae = np.mean(np.abs((output - target)))
            mape = np.mean(np.abs((output - target)/target))
        return {'mae': mae, 'mape': mape, 'rho': rho, 'P_rho': p}
    else:
        # AU-PRC vs. random (AU-PRC adjusted)
        if len(output.shape) == 1:
            target = target.unsqueeze(1)
        auprc_model = auprc(output, target, nan20=nan20)
        random_clf = torch.distributions.normal.Normal(0, 1)
        auprc_random = 0.
        for n in range(n_trials):
            random_output = random_clf.sample((output.shape[0], output.shape[1]))
            if self.out_dim > 1:
                random_output = torch.softmax(random_output, dim=-1)
            else:
                random_output = torch.sigmoid(random_output)
            auprc_random += auprc(random_output, target, nan20=nan20)
        auprc_random = auprc_random / n_trials
        auprc_adj = (auprc_model - auprc_random) / (1 - auprc_random)

        # balanced acc 
        balanced_acc = smklmetrics.balanced_accuracy_score(target, output, adjusted=False)
        balanced_acc_adj = smklmetrics.balanced_accuracy_score(target, output, adjusted=True)
        return {'auprc_model': auprc_model, 'auprc_adj': auprc_adj, 'balanced_acc': balanced_acc, 'balanced_acc_adj': balanced_acc_adj}


# eval DL models
class eval_trained():
    def __init__(self, trainer, modelpkl=None, split='test', 
                 two_outputs=False,
                 out_file=None):
        self.trainer = trainer
        self.exp_name = '{}_{}'.format(self.trainer.exp, self.trainer.trial)
        self.modelpkl = modelpkl
        self.split = split
        self.device = torch.device('cpu')
        self.trainer.model = self.trainer.model.to(self.device)
        self.trainer.model.eval()
        self.two_outputs = two_outputs # signal whether weights/attn/embedding also output
        self.get_model_output()
        self.eval_performance = eval_output(self.yhat, self.y, tasktype=trainer.data.tasktype)
        if out_file is not None:
            self.output_results(out_file)
    
    def get_model_output(self):
        self.results = pd.DataFrame()
        
        # data
        if self.split=='test': # switch for set to analyze
            dataloader = self.trainer.data.test_dl
        elif self.split=='train':
            dataloader = self.trainer.data.train_dl
        elif self.split=='val':
            dataloader = self.trainer.data.val_dl
        else:
            print('Must evaluate one of train/test/val split')
            
        # model
        if self.modelpkl is not None:
            self.trainer.model.load_state_dict(
                torch.load(self.modelpkl, map_location=self.device))
        self.trainer.model.eval()
        dataloader.num_workers = 1
        for i, batch in enumerate(dataloader):
            x, y, idx = batch['x'], batch['y'], batch['id']
            tic = time.time()

            if self.two_outputs:
                output, addl_out = self.trainer.model(x, addl_out=True)
            else:
                output = self.trainer.model(x)
            if self.trainer.data.tasktype == 'regression':
                output = output.squeeze()
            if i==0:
                y_total = y.detach()
                idx_total = idx
                yhat_total = output.detach()
                if self.two_outputs:
                    out2_total = addl_out.detach()
            else:
                y_total = torch.cat((y_total, y.detach()), dim=0)
                idx_total = idx_total + idx
                yhat_total = torch.cat((yhat_total, output.detach().reshape(-1, )), dim=0)
                if self.two_outputs:
                    out2_total = torch.cat((out2_total, addl_out.detach()), dim=0)            
                    
        # store
        self.y = y_total
        self.yhat = yhat_total
        self.id = idx_total
        if self.two_outputs:
            self.out2 = out2_total
        if 'MSELoss' in str(self.trainer.criterion.__class__) or 'NLLLoss' in str(self.trainer.criterion.__class__):
            self.loss_test = self.trainer.criterion(output, y).item()
        else:
            self.loss_test = self.trainer.criterion(output, y, self.trainer.model.parameters()).item()
        for i, k in enumerate(self.id):
            dt = pd.Series(self.trainer.data.data['data'][k]['md']).T
            dt['id'] = k
            dt['y'] = self.y[i].item()
            dt['yhat'] = self.yhat[i].item()
            self.results = self.results.append(dt, ignore_index=True)
            
    def output_results(self, file):
        dt = pd.DataFrame({'exp_trial':self.exp_name, 
                           'y':None, 'yhat':None, 
                           'loss':self.loss_test}, index=[0])
        dt.at[0, 'y'] = self.y
        dt.at[0, 'yhat'] = self.yhat
        for k in self.eval_performance.keys():
            dt[k] = self.eval_performance[k]
        
        if os.path.exists(file):
            dt.to_csv(file, mode='a', header=True)
        else:
            dt.to_csv(file)
            
def p_encoder(p):
    if p > 0.05:
        label = '' # n.s.
    elif p <= 0.001:
        label = '***'
    elif p <= 0.05 and p > 0.01:
        label = '*'
    elif p <= 0.01 and p > 0.001:
        label = '**'
    else: 
        label = 'Unclassified'
    return label

def summarize_long_table(results_longtab, metrics=['MAE', 'Rho'], group='Model', out_file=None):
    '''
    Arguments:
      results_longtab (pd.DataFrame): assumes that replicates are indicated in a trial col
        but otherwise repeated according to the group colname
    '''
    from scipy.stats import ttest_ind
    summary = pd.DataFrame()
    
    for g in results_longtab[group].unique():
        summary.loc[g, group] = g
        for m in metrics:
            temp = {}
            others = [gg for gg in results_longtab[group].unique() if gg!=g]
            a = results_longtab.loc[results_longtab[group]==g, m].to_numpy()
            summary.loc[g, m] = '{:.2f} ({:.2f})'.format(np.mean(a), np.std(a))
            for gg in others:
                b = results_longtab.loc[results_longtab[group]==gg, m].to_numpy()
                statistic, p = ttest_ind(a, b)
                temp['v.{}'.format(gg)] = (np.max(a) - np.max(b), p)
            # only retain min
            k2keep = min(temp, key=temp.get)
            summary.loc[g, 'Top-1 diff ({})'.format(m)] = '{:.2f} ({})'.format(temp[k2keep][0], k2keep)
            summary.loc[g, 'P ({})'.format(m)] = '{:.2e}{} ({})'.format(temp[k2keep][1], p_encoder(temp[k2keep][1]), k2keep)
    if out_file is not None:
        summary.to_csv(out_file)
    return summary
            

# if __name__ == '__main__':
#     mode = 'weighted_AU-PRC'
#     nan20 = False
#     p_class=[
#             [0.5, 0.4, 0.1],
#             [0.5, 0.49, 0.01],
#             [0.5, 0.499, 0.001],
#             [0.8, 0.1, 0.1],
#             [0.9, 0.09, 0.01],
#             [0.9, 0.099, 0.001]
#             ]
#     # p_class = [[0.5], [0.6], [0.7], [0.8], [0.9], [99/100], [999/1000]]
#     res_multi = eval_evalmetric(p_class=p_class).experiment(verbose=False, mode=mode, nan20=nan20)
#     fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#     p = sns.boxplot(x='P_class', y=mode, data=res_multi, ax=ax)
#     p.set_xticklabels(p.get_xticklabels(), rotation=45)
#     fig.tight_layout()
#     plt.show()
