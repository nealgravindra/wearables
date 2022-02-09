# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import time
import pickle
import datetime

import sys
sys.path.append('/home/ngrav/project/')
from wearables.scripts import utils as wearutils
from wearables.scripts import data as weardata
from wearables.scripts import train as weartrain
from wearables.scripts import eval_ as weareval
from wearables.scripts import model as wearmodels
from wearables.scripts import DTW as weardtw

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.regression.compose._ensemble import ComposableTimeSeriesForestRegressor
from sktime.classification.compose import ComposableTimeSeriesForestClassifier

plt.rc('font', size = 9)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['legend.frameon']=False
plt.rcParams['axes.grid']=False
plt.rcParams['legend.markerscale']=1
plt.rcParams['savefig.dpi'] = 600
sns.set_style("ticks")

class dataloader():
    '''Load data into numpy data types and specify number of folds. If kfold=1 or 0, then single val set returned.
    '''
    def __init__(self, target_name='GA', 
                 kfold=5, prop_trainset=1., 
                 include_lux=False, 
                 filename='/home/ngrav/data/wearables/processed/MOD1000_modeldata.pkl',
                 pp_md_fp='/home/ngrav/project/wearables/results/md_v522_220124.csv'):
        self.rawdata_file = filename
        self.data = self.load_preproced(filename)
        self.target_name = target_name
        self.kfold = kfold
        self.prop_trainset = prop_trainset
        self.include_lux = include_lux
        if pp_md_fp is not None:
            self.md = pd.read_csv(pp_md_fp, index_col=0)
        else:
            self.md = None
        self.ids = list(self.data['IDs'].keys())
        
        # split data and get dataloaders
        self.split_data()
        self.Xy_train, self.Xy_test = self.get_Xy(self.target_name)
        
    def load_preproced(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            f.close()
        return data
    
    def split_data(self, train_ratio=0.8):
        pids = np.unique([i.split('_')[0] for i in self.data['IDs'].keys()])
        train_pids = np.random.choice(pids, int(len(pids)*train_ratio*self.prop_trainset), replace=False)
        test_pids = [i for i in pids if i not in train_pids]
        if self.kfold <= 1: # create single val set
            val_pids = np.random.choice(test_pids, int(len(test_pids)*0.5), replace=False)
            # no overlap between val/test? or merge val into test?
            # test_pids = [i for i in test_pids if i not in val_pids]
            self.val_ids = [i for i in self.data['IDs'].keys() if i.split('_')[0] in val_pids]
        else:
            self.train_ids = np.array_split(train_pids, self.kfold)
            for ii, fold in enumerate(self.train_ids):
                self.train_ids[ii] = [i for i in self.data['IDs'].keys() if i.split('_')[0] in self.train_ids[ii]]
        self.test_ids = [i for i in self.data['IDs'].keys() if i.split('_')[0] in test_pids]
        return None

    def Xy_from_id(self, ids, target_name):
        for i, k in enumerate(ids):
            if self.include_lux:
                x = np.concatenate((self.data['data'][k]['activity'].to_numpy(dtype=np.float32)[:-1].reshape(-1, 1), 
                                    self.data['data'][k]['light'].to_numpy(dtype=np.float32)[:-1].reshape(-1, 1)), 
                                    1) 
            else:
                x = self.data['data'][k]['activity'].to_numpy(dtype=np.float32)[:-1]
            if i==0:
                X = np.zeros(shape=(len(ids), *x.shape))
                y = np.zeros(shape=(len(ids), ))
            X[i] = x
            if target_name == 'GA':
                y[i] = np.float32(k.split('_')[-1])
            else:
                if self.md is None:
                    y[i] = np.float32(self.data['data'][k]['md'][target_name])
                else:
                    y[i] = np.float32(self.md.loc[k, target_name].item())
                    
        # define in terms of category class
        try:
            enc = self.data['cat_class_enc'][target_name]
        except KeyError:
            enc = None
        if enc is not None and len(enc.keys()) > 2:
            # force to wide 
            y_wide = np.zeros(shape=(y.shape[0], len(enc.keys())), dtype=np.float32)
            y_wide[np.arange(y.shape[0]), np.array(y, dtype=int)] = 1.
            y = y_wide # force multiclass classification, check cat_class_enc for meaning
        return X, y

    def get_Xy(self, target_name):
        # train, if CV, dict with [(X_train, y_train), (X_val, y_val)] else (X_train, y_train)
        Xy_test = self.Xy_from_id(self.test_ids, target_name)
        if self.kfold > 1:
            Xy_train = {}
            Xy_cv = {}
            for k, fold_ids in enumerate(self.train_ids):
                Xy_cv[k] = (self.Xy_from_id(fold_ids, target_name))
            # now concatenate others, shifting single to others
            for kfold in Xy_cv.keys():
                for i, kk in enumerate([k for k in Xy_cv.keys() if k != kfold]):
                    X, y = Xy_cv[kk]
                    if i==0:
                        X_cv, y_cv = X.copy(), y.copy()
                    else:
                        X_cv = np.concatenate((X_cv, X), 0)
                        y_cv = np.concatenate((y_cv, y), 0)
                Xy_train[kfold] = [(X_cv, y_cv), Xy_cv[kfold]]
            return Xy_train, Xy_test
        else:
            Xy_train = self.Xy_from_id(self.train_ids, target_name)
            Xy_val = self.Xy_from_id(self.val_ids, target_name)
            return Xy_train, Xy_val, Xy_test
        
class kNN():
    def __init__(self, n_trials=1, target_name='GA', tasktype='regression'):
        self.n_trials = n_trials
        self.target_name = target_name
        self.tasktype = tasktype
    
        
    def load_data(self):
        return dataloader(target_name=self.target_name)
        
    def fit(self, data):
        if data.kfold > 1:
            cv_eval = {}
            for k, cv_fold in enumerate(data.Xy_train.keys()):
                [(X_train, y_train), (X_val, y_val)] = data.Xy_train[cv_fold]
                kNN = KNeighborsRegressor(n_jobs=16) if self.tasktype=='regression' else KNeighborsClassifier(n_jobs=16)
                kNN.fit(X_train, y_train)
                eval_metrics = weareval.eval_output(kNN.predict(X_val), y_val, tasktype=self.tasktype)
                cv_eval[cv_fold] = {'model': kNN, 
                                    # 'data': [(X_train, y_train), (X_val, y_val)], # store just IDs?
                                    'metric': eval_metrics['mae'] if self.tasktype=='regression' else eval_metrics['balanced_acc_adj'],
                                    'metrics': eval_metrics}
            # retain only best model
            tmp = {cv_fold:cv_eval[cv_fold]['metric'] for cv_fold in cv_eval.keys()}
            bst_fold = min(tmp, key=tmp.get) if self.tasktype=='regression' else max(tmp, key=tmp.get)
            self.kNN = cv_eval[bst_fold]['model']
            return {'model': self.kNN, 'metrics': cv_eval[bst_fold]['metrics']}
        else:
            X_train, y_train = data.Xy_train
            X_val, y_val = data.Xy_val
            self.kNN = KNeighborsRegressor(n_jobs=12) if self.tasktype=='regression' else KNeighborsClassifier(n_jobs=12)
            self.kNN.fit(X_train, y_train)
            eval_metrics = weareval.eval_output(self.kNN.predict(X_val), y_val, tasktype=self.tasktype)
            return {'model': self.kNN, 'metrics': eval_metrics}
    
    def eval_test(self, data):
        X_test, y_test = data.Xy_test
        eval_metrics = weareval.eval_output(self.kNN.predict(X_test), y_test, tasktype=self.tasktype)
        return eval_metrics
    
    def del_model(self):
        del self.kNN
    
    def run_trials(self, verbose=False):
        if verbose:
            t_start = time.time()
            print('Starting kNN trials; predict {}'.format(self.target_name))
        results = {}
        for n in range(self.n_trials):
            if verbose:
                tic = time.time()
            data = self.load_data()
            self.fit(data)
            results[n] = self.eval_test(data)
            self.del_model()
            if verbose:
                print('  finished trial {} in {:.2f}-s\t{:.1f}-min elapsed'.format(n, time.time()-tic, (time.time()-t_start)/60))
        return results
    
class TSRF():
    def __init__(self, n_trials=1, target_name='GA', tasktype='regression'):
        self.n_trials = n_trials
        self.target_name = target_name
        self.tasktype = tasktype
    
        
    def load_data(self):
        return dataloader(target_name=self.target_name, )
        
    def fit(self, data):
        if data.kfold > 1:
            cv_eval = {}
            for k, cv_fold in enumerate(data.Xy_train.keys()):
                [(X_train, y_train), (X_val, y_val)] = data.Xy_train[cv_fold]
                X_train, X_val = from_2d_array_to_nested(X_train), from_2d_array_to_nested(X_val)
                tsf = ComposableTimeSeriesForestRegressor(
                    n_jobs=-1) if self.tasktype=='regression' else ComposableTimeSeriesForestClassifier(
                    n_jobs=-1)
                tsf.fit(X_train, y_train)
                eval_metrics = weareval.eval_output(tsf.predict(X_val), y_val, tasktype=self.tasktype)
                cv_eval[cv_fold] = {'model': tsf, 
                                    # 'data': [(X_train, y_train), (X_val, y_val)], # store just IDs?
                                    'metric': eval_metrics['mae'] if self.tasktype=='regression' else eval_metrics['balanced_acc_adj'],
                                    'metrics': eval_metrics}
            # retain only best model
            tmp = {cv_fold:cv_eval[cv_fold]['metric'] for cv_fold in cv_eval.keys()}
            bst_fold = min(tmp, key=tmp.get) if self.tasktype=='regression' else max(tmp, key=tmp.get)
            self.tsf = cv_eval[bst_fold]['model']
            return {'model': self.tsf, 'metrics': cv_eval[bst_fold]['metrics']}
        else:
            X_train, y_train = data.Xy_train
            X_val, y_val = data.Xy_val
            X_train, X_val = from_2d_array_to_nested(X_train), from_2d_array_to_nested(X_val)
            self.tsf = ComposableTimeSeriesForestRegressor(
                n_jobs=-1) if self.tasktype=='regression' else ComposableTimeSeriesForestClassifier(
                n_jobs=-1)
            self.tsf.fit(X_train, y_train)
            eval_metrics = weareval.eval_output(self.tsf.predict(X_val), y_val, tasktype=self.tasktype)
            return {'model': self.tsf, 'metrics': eval_metrics}
    
    def eval_test(self, data):
        X_test, y_test = data.Xy_test
        eval_metrics = weareval.eval_output(self.tsf.predict(X_test), y_test, tasktype=self.tasktype)
        return eval_metrics
    
    def del_model(self):
        del self.tsf
    
    def run_trials(self, verbose=False):
        if verbose:
            t_start = time.time()
            print('Starting TSRF trials; predict {}'.format(self.target_name))
        results = {}
        for n in range(self.n_trials):
            if verbose:
                tic = time.time()
            data = self.load_data()
            self.fit(data)
            results[n] = self.eval_test(data)
            self.del_model()
            if verbose:
                print('  finished trial {} in {:.2f}-s\t{:.1f}-min elapsed'.format(n, time.time()-tic, (time.time()-t_start)/60))
        return results
    
# statistical tests
def pd_chisq(df, feat, groupby='Error group'):
    obs = md.groupby([groupby, feat]).size().unstack(fill_value=0)
    chi2, p, dof, expected = chi2_contingency(obs)
    return p, obs, expected

def pd_kruskalwallis(df, feat, groupby='Error group'):
    size = []
    for i, g in enumerate(df[groupby].unique()):
        dt = df.loc[df[groupby]==g, feat].to_numpy()
        size.append(dt.shape[0])
        if i==0:
            X = dt
        else:
            X = np.concatenate((X, dt))
    X = np.split(X, np.cumsum(size[:-1]))
    statistic, p = kruskal(*X)
    return p, X

def stat_err_analysis(md, voi2pred, model=None, 
                      groupby='Error group', 
                      n_trials=1,
                      verbose=True,
                      out_file=None):
    '''Extract Pval from non-parametric test for cat/cont vars and predictability across metadata
    
    Arguments:
      md (pd.DataFrame): metadata where index is unique identifier to call dataset
      voi2pred (dict): specify column in md to predict as key; value is 'categorical' or 'continuous' to 
        trigger appropriate test and model
      
    '''
    res = dict()
    if verbose:
        if model is not None:
            print('Starting stat and predictability on {} vars'.format(len(list(voi2pred.keys()))))
        else:
            print('Starting stats on {} vars'.format(len(list(voi2pred.keys()))))
        tic = time.time()
    for i, (k, v) in enumerate(voi2pred.items()):
        if v == 'continuous':
            p, obs, exp = pd_chisq(md, k, groupby=groupby)
            res[k] = {'p': p, 'obs': obs, 'exp': exp}
        else:
            p, X = pd_kruskalwallis(md, k, groupby=groupby)
            #quants = {'q{}'.format(i):md.groupby(groupby)[col].quantile(i).to_dict() for i in [0.25, 0.5, 0.75]}
            res[k] = {'p': p, 'X': X}
        if model is not None:
            res[k]['predictability'] = model(target_name=k, tasktype='regression' if v=='continuous' else 'classification', n_trials=n_trials).run_trials()
        if verbose:
            print('  through {} vars ({}) in {:.0f}-s'.format(i+1, k, time.time() - tic))
    if out_file is not None:
        with open(out_file, 'wb') as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
    return res

def goodmanKruskalgamma(data, ordinal1, ordinal2):
    # REF: https://colab.research.google.com/drive/1w1t7T67eKLoLzXfoRv4ZG6DsSK4q0UQK#scrollTo=wlWJK2B8DVMN
    from scipy.stats import norm
    myCrosstable = pd.crosstab(data[ordinal1], data[ordinal2])
    
#     myCrosstable = myCrosstable.reindex(orderLabels1)
            
#     if orderLabels2 == None:
#         myCrosstable = myCrosstable[orderLabels1]
#     else:
#         myCrosstable = myCrosstable[orderLabels2]

    nRows = myCrosstable.shape[0]
    nCols = myCrosstable.shape[1]
    
    
    C = [[0 for x in range(nCols)] for y in range(nRows)] 

    # top left part
    for i in range(nRows):
        for j in range(nCols):
            h = i-1
            k = j-1        
            if h>=0 and k>=0:            
                for p in range(h+1):
                    for q in range(k+1):
                        C[i][j] = C[i][j] + list(myCrosstable.iloc[p])[q]

    # bottom right part                    
    for i in range(nRows):
        for j in range(nCols):
            h = i+1
            k = j+1        
            if h<nRows and k<nCols:            
                for p in range(h, nRows):
                    for q in range(k, nCols):
                        C[i][j] = C[i][j] + list(myCrosstable.iloc[p])[q]
                        
    D = [[0 for x in range(nCols)] for y in range(nRows)] 

    # bottom left part
    for i in range(nRows):
        for j in range(nCols):
            h = i+1
            k = j-1        
            if h<nRows and k>=0:            
                for p in range(h, nRows):
                    for q in range(k+1):
                        D[i][j] = D[i][j] + list(myCrosstable.iloc[p])[q]

    # top right part                    
    for i in range(nRows):
        for j in range(nCols):
            h = i-1
            k = j+1        
            if h>=0 and k<nCols:            
                for p in range(h+1):
                    for q in range(k, nCols):
                        D[i][j] = D[i][j] + list(myCrosstable.iloc[p])[q]

    P = 0
    Q = 0
    for i in range(nRows):
        for j in range(nCols):
            P = P + C[i][j] * list(myCrosstable.iloc[i])[j]
            Q = Q + D[i][j] * list(myCrosstable.iloc[i])[j]
               
    GKgamma = (P - Q) / (P + Q)
    
    # pval calc
    
#     if abs(GKgamma) < .10:
#         qual = 'Negligible'
#     elif abs(GKgamma) < .20:
#         qual = 'Weak'
#     elif abs(GKgamma) < .40:
#         qual = 'Moderate'
#     elif abs(GKgamma) < .60:
#         qual = 'Relatively strong'
#     elif abs(GKgamma) < .80:
#         qual = 'Strong'        
#     else:
#         qual = 'Very strong'
    
#     n = myCrosstable.sum().sum()
    
#     Z1 = GKgamma * ((P + Q) / (n * (1 - GKgamma**2)))**0.5
    
#     forASE0 = 0
#     forASE1 = 0
#     for i in range(nRows):
#         for j in range(nCols):
#             forASE0 = forASE0 + list(myCrosstable.iloc[i])[j] * (Q * C[i][j] - P * D[i][j])**2
#             forASE1 = forASE1 + list(myCrosstable.iloc[i])[j] * (C[i][j] - D[i][j])**2

#     ASE0 = 4 * (forASE0)**0.5 / (P + Q)**2
#     ASE1 = 2 * (forASE1 - (P - Q)**2 / n)**0.5 / (P + Q)        
#     Z2 = GKgamma / ASE0
#     Z3 = GKgamma / ASE1
    
#     p1 = norm.sf(Z1)
#     p2 = norm.sf(Z2)
#     p3 = norm.sf(Z3)
    
#     zvalues = [Z1] + [Z2] + [Z3]
#     pvalues = [p1] + [p2] + [p3]
            
    return GKgamma# (GKgamma,qual), zvalues, pvalues

def tabular_corrnet(md, mdpred_voi):
    '''Calculate correlations (will need to take absolute value) for cont-cont using Spearman's rho,
         cont-cat using logistic regression on the categorical (max of balanced acc, adj., i.e., Youden's J),
         and Goodman Kruskal gamma for cat-cat variable comparisons.
    
    Arguments:
      md (pd.DataFrame): of mixed categorical and continuous variables. Must have a split column specifying train and
        test splits so logistic regression can be run
      mdpred_voi (dict): specify name of column in md as a key and the type of var as a value, accepts 'continuous' or 'categorical' 
        values in order to trigger the appropriate analysis.
        
    Returns:
      pd.DataFrame, nx.Graph
    '''
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    import sklearn.metrics as sklmetrics
    from sklearn.exceptions import ConvergenceWarning
    variables = list(mdpred_voi.keys())
    G = nx.Graph()
    G.add_nodes_from(variables)
    df = pd.DataFrame(index=variables, columns=variables)
    for i in range(len(variables)):
        for j in range(len(variables)):
            if i<=j:
                continue
            else:
                k, kk = variables[i], variables[j]
                v, vv = mdpred_voi[k], mdpred_voi[kk]
                if v == 'continuous' and vv == 'continuous':
                    # spearman's rho 
                    rho, p = spearmanr(md[k], md[kk])
                    df.loc[k, kk] = rho
                    df.loc[kk, k] = rho
                elif (v == 'continuous' and vv == 'categorical') or (v == 'categorical' and vv == 'continuous'):
                    # logistic regression
                    contvar = k if v == 'continuous' else kk
                    catvar = k if v == 'categorical' else kk
                    X_train = md.loc[md['split']=='train', contvar].to_numpy(dtype=np.float64).reshape(-1, 1)
                    X_test = md.loc[md['split']=='test', contvar].to_numpy(dtype=np.float64).reshape(-1, 1)
                    y_train = md.loc[md['split']=='train', catvar]
                    y_test = md.loc[md['split']=='test', catvar]
                    if len(y_train.unique()) < 1:
                        print(f"only one val found for {catvar}")
                    elif len(y_train.unique()) > 2:
                        y_train = y_train.to_numpy(dtype=int)
                        y_test = y_test.to_numpy(dtype=int)
                        y_train_wide = np.zeros((y_train.shape[0], len(np.unique(y_train))), dtype=int)
                        y_test_wide = np.zeros((y_test.shape[0], len(np.unique(y_train))), dtype=int)
                        y_train_wide[np.arange(y_train.shape[0]), y_train] = 1
                        y_test_wide[np.arange(y_test.shape[0]), y_test] = 1

                        y_train = y_train_wide
                        del y_train_wide
                        y_test = y_test_wide 
                        del y_test_wide
                    else:
                        y_train = y_train.to_numpy(dtype=int).reshape(-1, 1)
                        y_test = y_test.to_numpy(dtype=int).reshape(-1, 1)
                    balanced_acc = [] # Youden's J
                    for j in range(y_train.shape[1]):
                        lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.1)
                        lr.fit(X_train, y_train[:, j])
                        balanced_acc.append(
                            sklmetrics.balanced_accuracy_score(y_test[:, j], lr.predict(X_test), adjusted=True)
                        )
                    df.loc[k, kk] = np.max(balanced_acc) # max agg?
                    df.loc[kk, k] = np.max(balanced_acc) # max agg?
                else:
                    # cat v. cat
                    GKgamma = wearerr.goodmanKruskalgamma(md, k, kk)
                    df.loc[k, kk] = GKgamma 
                    df.loc[kk, k] = GKgamma
                G.add_edge(k, kk, weight=np.abs(df.loc[k, kk]))
    return df, G

if __name__ == '__main__':
    # load data
    pfp = '/home/ngrav/project/wearables/results/'
    mfp = '/home/ngrav/project/wearables/model_zoo'

    # filepaths to bst or pre-processed md with calculated metrics 
    pp_md_fp = os.path.join(pfp, 'md_v522_220124.csv')
    bst_trainer = os.path.join(mfp, 'trainer_itv52_InceptionTime_GA5.pkl')
    bst_modelpkl = os.path.join(mfp, '213-itv52_InceptionTime_GA5.pkl')
    
    # load up to date md
    md = pd.read_csv(pp_md_fp, index_col=0)
    
    # threshold for error groups
    threshold = 10 # wks
    md['Error group'] = 'lt{}wks'.format(threshold)
    md.loc[(md['error'] >= threshold), 'Error group'] = 'Higher-than-actual'
    md.loc[(md['error'] <= -threshold), 'Error group'] = 'Lower-than-actual'

    
    if False:
        trainer = loadpkl(bst_trainer)

    if False:
        # call eval to get output, modify for return_hidden = True option
        evaluation = weareval.eval_trained(trainer, split='train',
                                           modelpkl=bst_modelpkl,
                                           two_outputs=True)
        
    # grab list of vars to predict (mdpred_voi)
    from wearables.scripts.md_specification import mdpred_voi
    
    total_t = time.time()
    
#     tic = time.time()
#     res = stat_err_analysis(md, 
#                             mdpred_voi, 
#                             out_file=os.path.join(pfp, 'md_predictability_tsrf_10wk.pkl'), 
#                             model=TSRF)
#     print('Done w/10wk TSRF in {:.1f}-m\t({:.1f}-min elapsed)'.format( 
#         (time.time() - tic)/60, (time.time() - total_t)/60))
    
    tic = time.time()
    res = stat_err_analysis(md, 
                            mdpred_voi, 
                            out_file=os.path.join(pfp, 'md_predictability_knn_10wk.pkl'), 
                            model=kNN,
                            n_trials=10)
    print('Done w/10wk kNN in {:.1f}-m\t({:.1f}-min elapsed)'.format( 
        (time.time() - tic)/60, (time.time() - total_t)/60))
    
    # change threshold
    # threshold for error groups
    threshold = 8 # wks
    md['Error group'] = 'lt{}wks'.format(threshold)
    md.loc[(md['error'] >= threshold), 'Error group'] = 'Higher-than-actual'
    md.loc[(md['error'] <= -threshold), 'Error group'] = 'Lower-than-actual'
    
#     tic = time.time()
#     res = stat_err_analysis(md, 
#                             mdpred_voi, 
#                             out_file=os.path.join(pfp, 'md_predictability_tsrf_8wk.pkl'), 
#                             model=TSRF)
#     print('Done w/8wk TSRF in {:.1f}-m\t({:.1f}-min elapsed)'.format( 
#         (time.time() - tic)/60, (time.time() - total_t)/60))
    
    tic = time.time()
    res = stat_err_analysis(md, 
                            mdpred_voi, 
                            out_file=os.path.join(pfp, 'md_predictability_knn_8wk.pkl'), 
                            model=kNN,
                            n_trials=10)
    print('Done w/8wk kNN in {:.1f}-m\t({:.1f}-min elapsed)'.format( 
        (time.time() - tic)/60, (time.time() - total_t)/60))
        
    