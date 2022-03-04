import os
import time
import datetime
import pandas as pd
import numpy as np
import random
from imblearn.over_sampling import SMOTE

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
                    GKgamma = goodmanKruskalgamma(md, k, kk)
                    df.loc[k, kk] = GKgamma 
                    df.loc[kk, k] = GKgamma
    return df

# def colVall_corr(md, coi, mdpred_voi, groupby=None, verbose=True):
#     '''Correlate one column with all others for cont-cont using Spearman's rho,
#          cont-cat using logistic regression on the categorical (max of balanced acc, adj., i.e., Youden's J),
#          and Goodman Kruskal gamma for cat-cat variable comparisons.
    
#     Arguments:
#       md (pd.DataFrame): of mixed categorical and continuous variables. Must have a split column specifying train and
#         test splits so logistic regression can be run
#       coi (str): specify column name with which to correlate all in mdpred_voi with (self-corr ignored)
#       mdpred_voi (dict): specify name of column in md as a key and the type of var as a value, accepts 'continuous' or 'categorical' 
#         values in order to trigger the appropriate analysis.
#       groupby
        
#     Returns:
#       pd.Series
#     '''
#     from scipy.stats import spearmanr
#     from sklearn.linear_model import LogisticRegression
#     import sklearn.metrics as sklmetrics
#     from sklearn.exceptions import ConvergenceWarning
#     variables = [i for i in list(mdpred_voi.keys()) if i!=coi]
#     grps = ['all']
#     if groupby is not None:
#         grps += list(md[groupby].unique())
#     df = pd.DataFrame(index=variables, columns=grps)
#     for i in grps:
#         if verbose:
#             group_t = time.time()
#             print(f"Starting grp: {i}")
#         dt = md if i=='all' else md.loc[md[groupby]==i, :]
#         dt = dt.reset_index()
#         # fast recreation of splits from df 
#         split = ['train']*int(0.8*len(dt.index)) + ['test']*(len(dt.index) - int(0.8*len(dt.index)))
#         random.shuffle(split)
#         dt['split'] = split
#         for j in range(len(variables)):
#             k, kk = coi, variables[j]
#             v, vv = mdpred_voi[k], mdpred_voi[kk]
#             if v == 'continuous' and vv == 'continuous':
#                 # spearman's rho 
#                 rho, p = spearmanr(dt[k], dt[kk])
#                 df.loc[kk, i] = rho
#             elif (v == 'continuous' and vv == 'categorical') or (v == 'categorical' and vv == 'continuous'):
#                 # logistic regression
#                 contvar = k if v == 'continuous' else kk
#                 catvar = k if v == 'categorical' else kk
#                 X_train = dt.loc[dt['split']=='train', contvar].to_numpy(dtype=np.float64).reshape(-1, 1)
#                 X_test = dt.loc[dt['split']=='test', contvar].to_numpy(dtype=np.float64).reshape(-1, 1)
#                 y_train = dt.loc[dt['split']=='train', catvar]
#                 y_test = dt.loc[dt['split']=='test', catvar]
#                 if len(y_train.unique()) < 1:
#                     print(f"only one val found for {catvar}")
#                 elif len(y_train.unique()) > 2:
#                     try:
#                         y_train = y_train.to_numpy(dtype=int)
#                     except ValueError:
#                         encoding = np.sort(y_train.unique()) # e.g., ['Evening', 'Morning', 'NA'] = [0, 1, 2]
#                         y_train = y_train.map({orig_key:iii for iii, orig_key in enumerate(encoding)}).to_numpy(dtype=int)
#                     try:
#                         y_test = y_test.to_numpy(dtype=int)
#                     except ValueError:
#                         y_test = y_test.map({orig_key:iii for iii, orig_key in enumerate(encoding)}).to_numpy(dtype=int)
#                     y_train_wide = np.zeros((y_train.shape[0], len(np.unique(y_train))), dtype=int)
#                     y_test_wide = np.zeros((y_test.shape[0], len(np.unique(y_train))), dtype=int)
#                     try: # error comes for visit_num because it's categorical but not in [0, 1, 2,] rather [1, 2, 3] so needs mod (just drop it)
#                         y_train_wide[np.arange(y_train.shape[0]), y_train] = 1
#                     except IndexError: # missing a value, not properly encoded
#                         encoding = {orig_key:iii for iii, orig_key in enumerate(np.sort(np.unique(y_train)))}
#                         for orig_val in encoding.keys():
#                             y_train[y_train == orig_val] = encoding[orig_val]
#                             y_test[y_test == orig_val] = encoding[orig_val]
#                         y_train_wide[np.arange(y_train.shape[0]), y_train] = 1
#                         y_test_wide[np.arange(y_test.shape[0]), y_test] = 1
#                     y_train = y_train_wide
#                     del y_train_wide
#                     y_test = y_test_wide 
#                     del y_test_wide
#                 else:
#                     try:
#                         y_train = y_train.to_numpy(dtype=int).reshape(-1, 1)
#                     except ValueError: # not encoded yet
#                         encoding = np.sort(y_train.unique()) # e.g., ['Weeday', 'Weekend'] = [0, 1]
#                         y_train = y_train.map({k:i for i, k in enumerate(encoding)}).to_numpy(dtype=int).reshape(-1, 1)
#                     try:
#                         y_test = y_test.to_numpy(dtype=int).reshape(-1, 1)
#                     except ValueError:
#                         y_test = y_test.map({k:i for i, k in enumerate(encoding)}).to_numpy(dtype=int).reshape(-1, 1)
#                 balanced_acc = [] # Youden's J
#                 for jj in range(y_train.shape[1]):
#                     lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.1)
#                     try:
#                         lr.fit(X_train, y_train[:, jj])
#                     except ValueError: # only one class:
#                         return (X_train, y_train, kk)
#                         balanced_acc.append(np.nan)
#                         continue
#                     balanced_acc.append(
#                         sklmetrics.balanced_accuracy_score(y_test[:, jj], lr.predict(X_test), adjusted=True)
#                     )
#                 df.loc[kk, i] = np.max(balanced_acc) # max agg?
#             else:
#                 # cat v. cat
#                 GKgamma = goodmanKruskalgamma(dt, k, kk)
#                 df.loc[kk, i] = GKgamma 
#             if verbose and j % 50 == 0:
#                 print(f"  through {j+1} of {len(variables)} vars in {time.time() - group_t:.0f}-s")
#         if verbose:
#             print(f"\n  ... through grp {i} in {(time.time() - group_t)/60:.1f}-min")
#     return df

def colVall_corr(md, coi, mdpred_voi, groupby=None, verbose=True, sample_groups='record_id'):
    '''Correlate one column with all others for cont-cont using Spearman's rho,
         cont-cat using logistic regression on the categorical (max of balanced acc, adj., i.e., Youden's J),
         and Goodman Kruskal gamma for cat-cat variable comparisons.
    
    Arguments:
      md (pd.DataFrame): of mixed categorical and continuous variables. Must have a split column specifying train and
        test splits so logistic regression can be run
      coi (str): specify column name with which to correlate all in mdpred_voi with (self-corr ignored)
      mdpred_voi (dict): specify name of column in md as a key and the type of var as a value, accepts 'continuous' or 'categorical' 
        values in order to trigger the appropriate analysis.
      groupby
        
    Returns:
      pd.Series
    '''
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    import sklearn.metrics as sklmetrics
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import cross_val_score #GroupKFold
    from sklearn.preprocessing import label_binarize
    variables = [i for i in list(mdpred_voi.keys()) if i!=coi]
    grps = ['all']
    if groupby is not None:
        grps += list(md[groupby].unique())
    df = pd.DataFrame(index=variables, columns=grps)
    for i in grps:
        if verbose:
            group_t = time.time()
            print(f"Starting grp: {i}")
        dt = md if i=='all' else md.loc[md[groupby]==i, :]
        dt = dt.reset_index()
        for j in range(len(variables)):
            k, kk = coi, variables[j]
            v, vv = mdpred_voi[k], mdpred_voi[kk]
            if v == 'continuous' and vv == 'continuous':
                # spearman's rho 
                rho, p = spearmanr(dt[k], dt[kk])
                df.loc[kk, i] = rho
            elif (v == 'continuous' and vv == 'categorical') or (v == 'categorical' and vv == 'continuous'):
                # logistic regression
                contvar = k if v == 'continuous' else kk
                catvar = k if v == 'categorical' else kk
                X_train = dt.loc[:, contvar].to_numpy(dtype=np.float64).reshape(-1, 1)
                y_train = dt.loc[:, catvar]
                # convert to int
                if len(y_train.unique()) == 1:
                    print(f"\nonly one val for {catvar}")
                    print('... cannot fit one class only. Reconsider its inclusion. Skipping this var.')
                    continue
                else:
                    try:
                        y_train = y_train.to_numpy(dtype=int)
                        if len(np.unique(y_train)) > 2:
                            y_train = label_binarize(y_train, classes=np.sort(np.unique(y_train)))
                        else:
                            y_train = y_train.reshape(-1, 1)
                    except ValueError:
                        y_train = label_binarize(y_train, classes=np.sort(np.unique(y_train)))
                # SMOTE
                clf_metric = [] # AU-ROC
                for jj in range(y_train.shape[1]):
                    oversample = SMOTE(k_neighbors=3)
                    # print(f'kk: {kk}\tX_train: {X_train.shape}\ty_train: {y_train.shape}')
                    try:
                        X_train_mod, y_train_mod = oversample.fit_resample(X_train, y_train[:, jj])
                    except ValueError:
                        print("\n{}'s {}-th class cannot be computed. Too few n_samples. Skipping".format(kk, jj)) 
                        if verbose:
                            print('{} class frequencies:'.format(kk))
                            for jj in range(y_train.shape[1]):
                                print(f"j: {jj}\t0: {(y_train[:, jj]==0).sum()}\t1: {(y_train[:, jj]==1).sum()}")
                        continue
                    del oversample                
                
                    # model/eval
                    lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.1)
                    scores = cross_val_score(lr, X_train_mod, y_train_mod, cv=5, scoring='roc_auc')
                    clf_metric.append(np.nanmean(scores))
                try:
                    df.loc[kk, i] = np.nanmax(clf_metric) 
                except ValueError:
                    print("\n{}:{} association cannot be computed: no model completed".format(k, kk)) 
                    
            else:
                # cat v. cat
                GKgamma = goodmanKruskalgamma(dt, k, kk)
                df.loc[kk, i] = GKgamma 
            if verbose and j % 50 == 0:
                print(f"  through {j+1} of {len(variables)} vars in {time.time() - group_t:.0f}-s")
        if verbose:
            print(f"\n  ... through grp {i} in {(time.time() - group_t)/60:.1f}-min")
    return df