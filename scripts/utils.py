import os
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', size = 9)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['legend.frameon']=False
plt.rcParams['axes.grid']=False
plt.rcParams['legend.markerscale']=1
plt.rcParams['savefig.dpi'] = 600
sns.set_style("ticks")

class timer():
    def __init__(self):
        self.laps = []

    def start(self):
        self.tic = time.time()

    def stop(self):
        self.laps.append(time.time() - self.tic)
        return self.laps[-1]

    def sum(self):
        return sum(self.laps)

# fast model boot up
def load_IT(modelpkl, target='GA', eval_trainset=False):
    # get md and error for node color
    import sys
    sys.path.append('/home/ngr4/project')
    from wearables.scripts import train_v3 as weartrain
    
    trainer = weartrain.InceptionTime_trainer(exp='preload', target=target)
    res = trainer.eval_test(modelpkl, eval_trainset=eval_trainset)

    return trainer, res

def estimate_model_mem(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    return '{:.0f} MB'.format(mem/(1e6)) # in MB

def tensor_mem_size(a):
    size = (a.element_size() * a.nelement()) / (1e6) # MB
    return '{:.0f} MB'.format(size)


def tdiff_from24htime(start, end, nan2zero=True, minutes_not_h=True):
    '''
    Arguments:
      a (pd.Series): does not assume it's in pd.Datetime format
      b (pd.Series): does not assume it's in pd.Datetime format
      minutes_not_h (bool): [optional, Default=True] specify if you want 
        units in minutes (set True) or hours (set False)
    '''
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    assert all(start.index == end.index), 'must have alignment'
    d = pd.Series(index=start.index, dtype='float64')
    unit = 'timedelta64[m]' if minutes_not_h else 'timedelta64[h]'
    d.loc[start <= end] = (end - start).astype(unit)
    d.loc[start > end] = ((end + pd.Timedelta(24, unit='h')) - start).astype(unit)
    if nan2zero:
        d = d.fillna(value=0.)
    return d

def compare_v3tov1(raw_metadata, metadata_prime, plot=True, var=['PQSI', 'KPAS', 'EpworthSS', 'Edinburgh'], save_magic=None, 
                   cmap={'V3': '#AC514F', 'V1': '#5D5C61'}):
    from scipy.stats import mannwhitneyu
    res = {v:None for v in var}
    for v in var:
        a = raw_metadata['%s_3' % v].to_numpy()
        b = raw_metadata['%s_1' % v].to_numpy()
        if any(b==0):
            print('+1 offset for {}'.format(v))
        mean_diff = np.mean((a - b)/b) if not any(b==0) else np.mean(((a+1) - (b+1))/(b+1))# pct change, adj for linear scale
        _, p = mannwhitneyu(a, b)
        res[v] = (mean_diff, p)
        all_measurements = [metadata_prime[v].quantile(0.5), metadata_prime[v].quantile(0.25), metadata_prime[v].quantile(0.75)]
        v3 = [np.median(a), np.quantile(a, 0.25), np.quantile(a, 0.75)]
        v1 = [np.median(b), np.quantile(b, 0.25), np.quantile(b, 0.75)]
        print('{}:'.format(v))
        print('  all: {:.2f} ({:.2f} - {:.2f})\tv3: {:.2f} ({:.2f} - {:.2f})\t v1: {:.2f} ({:.2f} - {:.2f})'.format(
            *all_measurements, *v3, *v1))
        
        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(3.5, 2), gridspec_kw={'width_ratios': [3, 1]})
            ax[0].set_title(v)
            sns.distplot(metadata_prime[v], ax=ax[0], color='#FBC740')
            ax[0].set_xlabel('')
            ax[0].axes.get_yaxis().set_visible(False)
            dt = pd.DataFrame({'Percent change (%)': a, 'Visit': ['V3']*len(a)}).append(
                pd.DataFrame({'Percent change (%)': b, 'Visit': ['V1']*len(b)})
            )
            sns.boxplot(x="Visit", y="Percent change (%)", data=dt,
                        width=.6, palette=cmap, ax=ax[1])
            sns.stripplot(x="Visit", y="Percent change (%)", data=dt,
                          size=1, color=".3", linewidth=0, ax=ax[1], rasterized=True)
            ax[1].set_xlabel('')
            ax[1].set_ylabel('')
            ax[1].set_ylim([-1, 30])
            ax[1].set_yticks([0, 10, 20, 30])
            ax[1].set_yticklabels([0, '', '', 30])
            fig.tight_layout()
            
            if save_magic is not None:
                fig.savefig(save_magic + '_%s.pdf' % v, bbox_inches='tight', dpi=600)
    return res

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


def perm_test_oneVrest(md, n_iter=1000, n_sample=500, verbose=True):
    if verbose:
        tic = time.time()
    mean_sqdiff_rhos = {'obs': [], 'null': []}
    for i in range(n_iter):
        # obs
        idx_pos = list(md.loc[(md['split']=='test') & (md['Pre-term birth']==True)].sample(n_sample, replace=True).index)
        idx_neg = list(md.loc[~((md['split']=='test') & (md['Pre-term birth']==True))].sample(n_sample, replace=True).index)
        rho_pos, _ = spearmanr(md.loc[idx_pos, 'y'],
                               md.loc[idx_pos, 'yhat'])
        rho_neg, _ = spearmanr(md.loc[idx_neg, 'y'],
                               md.loc[idx_neg, 'yhat'])
        mean_sqdiff_rhos['obs'].append((rho_pos - rho_neg)**2)

        # null
        idx_pos = list(md.sample(n_sample, replace=True).index)
        idx_neg = list(md.sample(n_sample, replace=True).index)
        rho_pos, _ = spearmanr(md.loc[idx_pos, 'y'],
                               md.loc[idx_pos, 'yhat'])
        rho_neg, _ = spearmanr(md.loc[idx_neg, 'y'],
                               md.loc[idx_neg, 'yhat'])
        mean_sqdiff_rhos['null'].append((rho_pos - rho_neg)**2)
        
        if verbose and i % 100 == 0:
            print('through {} iter in {:.0f}-s'.format(i+1, time.time() - tic))
    p_est = (mean_sqdiff_rhos['obs'] > mean_sqdiff_rhos['null']) / n_iter
    return mean_sqdiff_rhos, p_est

def biaxial_fx(y_vars, data, x_var='GA', palette=None, hue='Pre-term birth', out_file=None):
    x = data[x_var]
    fig = plt.figure(figsize=(3.5, 4))
    for i, var in enumerate(y_vars):
        y = data[var]
        ax = fig.add_subplot(3, 2, i+1)
        ax.scatter(x, y, 
                   c=data[hue] if palette is None else data[hue].map(palette),
                   s=3, linewidth=0, alpha=0.8, rasterized=True)
        ax.set_ylabel(var)
        
        rho, p = spearmanr(x, y)
#         ax.text(0.1, 0.8, 'r={:.2f}{}'.format(rho, p_encoder(p)), transform=ax.transAxes, weight='bold')
        ax.set_title('r={:.2f}{}'.format(rho, p_encoder(p)))
        if i==4 or i==5:
            ax.set_xlabel('Actual GA')
    fig.tight_layout()
    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight', dpi=600)
        
def metric_list(stat_err_anal_res, mdpred_voi, 
                clf_metric='auprc_adj', clf_metric2='auprc_pctdiff', 
                reg_metric='rho', reg_metric2='mape',
                top1=False):
    out = {} # first is -log10 ( P ), second is metric specified
    for i, k in enumerate(stat_err_anal_res.keys()):
        if mdpred_voi[k] == 'continuous':
            metric_key = reg_metric
            metric2_key = reg_metric2
        else: 
            metric_key = clf_metric
            metric2_key = clf_metric2
        metric = []
        metric2 = []
        for kk in stat_err_anal_res[k]['predictability'].keys():
            m = stat_err_anal_res[k]['predictability'][kk][metric_key]
            if metric_key == 'auprc_adj':
                metric.append(m[0])
            else:
                metric.append(m)
            if metric2_key == 'auprc_pctdiff':
                m2 = (m[0] - m[1]) / m[1] 
                if np.isinf(m2):
                    m2 = np.nan
                    # clip?
#                 elif m2 < -1.:
#                     m2 = -1.
#                 elif m2 > 1.:
#                     m2 = +1.
                metric2.append(m2)
            elif metric2_key == 'mape':
                m2 = stat_err_anal_res[k]['predictability'][kk][metric2_key]
                m2 = 1 - m2 # want better. Clip it
                if np.isinf(m2):
                    m2 = np.nan
#                 elif m2 < -1.:
#                     m2 = -1.
#                 elif m2 > 1.:
#                     m2 = +1.
                metric2.append(m2)
            else:
                m2 = stat_err_anal_res[k]['predictability'][kk][metric2_key]
                metric2.append(m2)
        if metric_key == 'rho':
            metric = np.abs(metric)
        if metric2_key == 'rho':
            metric2 = np.abs(metric2)
        if top1:
            metric = np.max(metric)
            metric2 = np.max(metric2)
        else:
            metric = np.nanmean(metric)
            metric2 = np.nanmean(metric2)
        out[k] = (-np.log10(stat_err_anal_res[k]['p']), metric, metric2)
    return out


def goodmanKruskalgamma(data, ordinal1, ordinal2):
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
                    GKgamma = wearerr.goodmanKruskalgamma(md, k, kk)
                    df.loc[k, kk] = GKgamma 
                    df.loc[kk, k] = GKgamma
    return df

def range_scale(x, min_target=0, max_target=1):
    return ((x - np.min(x)) / (np.max(x) - np.min(x)))*(max_target - min_target) + min_target

def sigvars_per_cluster(metadata, voi, cluster_key='leiden', bonferonni_crct=True, verbose=True):
    '''Exclusive significance in cluster that is enriched categorically or has log2FC average >=0.5.'''
    def pval2sigvarlist(res, p_cutoff=0.001 / len(voi.keys()) if bonferonni_crct else 0.001, min_l2fc=0.5):
        filtered_res = {k:[] for k in res.keys()}
        for cid in res.keys():
            other_cids = [i for i in res.keys() if i!=cid]
            for var, val in res[cid].items():
                if (val[0] <= p_cutoff and not any([True if res[k][var][0] <= p_cutoff else False for k in other_cids])) and (isinstance(val[1], np.float32) and np.abs(val[1]) >= min_l2fc):
                    filtered_res[cid].append({'name': '{}_l2fc(c-rest)={:.2f}'.format(var, val[1]),
                                              'P_adj': val[0] / len(voi.keys()) if bonferonni_crct else val[0], 'log2fc(c-rest)': val[1]})
                elif (val[0] <= p_cutoff and not any([True if res[k][var][0] <= p_cutoff else False for k in other_cids])) and (isinstance(val[1], pd.DataFrame) and np.unravel_index(np.argmax(val[1].abs()), val[1].shape)[1] == 1):
                    idx = np.unravel_index(np.argmax(val[1].abs()), val[1].shape)
                    filtered_res[cid].append({'name': '{}={} enriched by {:.2f}-%'.format(var, val[1].index[idx[0]], 100*val[1].iloc[idx]),
                                         'P_adj': val[0] / len(voi.keys()) if bonferonni_crct else val[0], 'obs/exp-1': val[1]})
        return filtered_res
    from scipy.stats import chi2_contingency
    from scipy.stats import kruskal
    results = {c:{} for c in np.sort(metadata[cluster_key].unique())}
    # one-vs-rest scheme
    for i, c in enumerate(np.sort(metadata[cluster_key].unique())):
        metadata['cluster_part'] = (metadata[cluster_key] == c)
        for ii, v in enumerate(voi.keys()):
            if voi[v] == 'continuous':
                v_c = metadata.loc[metadata[cluster_key]==c, v]
                v_notc = metadata.loc[metadata[cluster_key]!=c, v]
                statistic, p = kruskal(v_c, v_notc)
                metric = np.log2(np.nanmean(v_c)) - np.log2(np.nanmean(v_notc)) # log2FC
                metric = np.float32(metric)
            else:
                obs = metadata.groupby([v, 'cluster_part']).size().unstack(fill_value=0)
                chi2, p, dof, expected = chi2_contingency(obs) # Fischer's?
                metric = ((obs / expected) - 1) # obs/expected ratio
            results[c][v] = ( p, metric )
    out = pval2sigvarlist(results)
    if verbose:
        for k in np.sort(list(out.keys())):
            for v in out[k]:
                print('cluster_id: {}, annotation: {}'.format(k, v['name'])) 
    return out


# more detailed analysis 
# compare expected vs observed in plot
def group_stat(df, col, include_var_pred_performance=False, norm=True,
                      groupby='Error group', 
                      omit_cols=['index', 'record_id', 'pid', 'Absolute Error', 'yGA', 'yhatGA'],
                      out_file=None):
    def pd_chisq(df, feat, groupby='Error group'):
        from scipy.stats import chi2_contingency
        obs = df.groupby([groupby, feat]).size().unstack(fill_value=0)
        chi2, p, dof, expected = chi2_contingency(obs)
        return p, obs, expected
    
    if df[col].dtype == object:
        p, obs, expected = pd_chisq(df, col, groupby=groupby)
        expected = pd.DataFrame(expected, 
                                index=obs.index.to_list(), 
                                columns=obs.columns.to_list())
        total = sum([v2 for k,v1 in obs.to_dict().items() for v2 in v1.values()])
        p_obs = obs.to_dict()
        p_exp = expected.to_dict()
        for k in p_obs.keys():
            p_obs[k] = {k:v/total for k,v in p_obs[k].items()}
            p_exp[k] = {k:v/total for k,v in p_exp[k].items()}
        out = {'Variable': col, 'P': p, 
               'Tasktype': 'Classification', 
               'Observed': obs.to_dict(),
               'Expected': expected.to_dict(),
               'pObserved': p_obs, 
               'pExpected': p_exp,}
    else:
        p = pd_kruskalwallis(df, col, groupby=groupby)
        out = {'Variable': col, 'P': p,
               'Tasktype': 'Classification', 
               'Observed': {'q{}'.format(i):df.groupby(groupby)[col].quantile(i).to_dict() for i in [0.25, 0.5, 0.75]}}
    return out

def extract_ratio(df, dfcolname, value, group, groupby='Error group', shuffle=False, randomize=False):
    '''ratio of obs/expected
    
    Arguments:
      value: cateogrical value
      group: specific group in groupby column
      shuffle: draw random samples of that error group to get an estimate of the metric
      randomize: compare obs to null dist for permutation test
    '''
    df = df.copy(deep=True)
    if shuffle:
        # turn off randomize
        dt = pd.DataFrame()
        for g in df[groupby].unique():
            dt = dt.append(df.loc[df[groupby]==g, :].sample((df[groupby]==g).sum(), replace=True), ignore_index=True)
        df = dt
    if randomize:
        df[groupby] = df[groupby].sample(frac=1, replace=True).to_list()
    sub_res = group_stat(df, dfcolname, groupby=groupby)
    obs = pd.DataFrame(sub_res['pObserved'])
    exp = pd.DataFrame(sub_res['pExpected'])
    factor = obs/exp
#     return factor
    return factor.loc[group, value]

def md_group_diffs(df, voi={'ptb_37wks': 'categorical',
                            'GA': 'continuous'}, 
                   groupby='Error group',
                   ratio_only=False,
                   out_file=None):
    
    def pd_chisq(df, feat, groupby='Error group'):
        from scipy.stats import chi2_contingency
        obs = df.groupby([groupby, feat]).size().unstack(fill_value=0)
        chi2, p, dof, expected = chi2_contingency(obs)
        return p, obs, expected
    
    def pd_kruskalwallis(df, feat, groupby='Error group'):
        from scipy.stats import kruskal
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
        return p
    
    # main block
    results = {g: {} for g in voi.keys()}
    for g, dtype in voi.items():

        if dtype == 'categorical':
            p, obs, expected = pd_chisq(df, g, groupby=groupby)
            expected = pd.DataFrame(expected, 
                                    index=obs.index.to_list(), 
                                    columns=obs.columns.to_list())
            total_obs = sum([v2 for k,v1 in obs.to_dict().items() for v2 in v1.values()])
            total_exp = sum([v2 for k,v1 in expected.to_dict().items() for v2 in v1.values()])
            p_obs = obs.to_dict()
            p_exp = expected.to_dict()
            for k in p_obs.keys():
                p_obs[k] = {k:v/total_obs for k,v in p_obs[k].items()}
                p_exp[k] = {k:v/total_exp for k,v in p_exp[k].items()}
            results[g] = {'Variable': g, 'P': p, 
                   'Tasktype': 'Classification', 
                   'Observed': obs.to_dict(),
                   'Expected': expected.to_dict(),
                   'pObserved': p_obs, 
                   'pExpected': p_exp,}
        else:
            p = pd_kruskalwallis(df, g, groupby=groupby)
            results[g] = {
                'Variable': g, 'P': p,
                'Tasktype': 'Classification', 
                'Observed': {'q{}'.format(i):df.groupby(groupby)[g].quantile(i).to_dict() for i in [0.25, 0.5, 0.75]},
                'Expected': {'q{}'.format(i):df[g].quantile(i) for i in [0.25, 0.5, 0.75]},
                'log2_grpVall' : {str(gg):np.log2(df.loc[df[groupby]==gg, g].mean()) - np.log2(df[g].mean()) for gg in df[groupby].unique()},
            }
    return results


def sample_metric(df, 
                  voi={'ptb_37wks': 'categorical','GA': 'continuous'}, 
                  value=None, n_samples=100, groupby='Error group'):
    '''ratio of obs/expected

    Arguments:
      value (dict): [optional, Default=None] provide dict with value to select from col of table
    '''
    import random
    dt = df.sample(n_samples, replace=True)
    grp = dt[groupby].to_list()
    random.shuffle(grp)
    dt['{}_shuffled'.format(groupby)] = grp
    results = {k: {} for k in voi.keys()}
    for i, (k, dtype) in enumerate(voi.items()):
        sub_res = md_group_diffs(dt, {k:voi[k]}, groupby=groupby)[k]
        sub_res_null = md_group_diffs(dt, {k:voi[k]}, groupby='{}_shuffled'.format(groupby))[k]
        if dtype=='categorical':
            # obs
            obs = pd.DataFrame(sub_res['pObserved'])
            exp = pd.DataFrame(sub_res['pExpected'])
            factor = obs/exp
            # null
            obs_null = pd.DataFrame(sub_res_null['pObserved'])
            exp_null = pd.DataFrame(sub_res_null['pExpected'])
            factor_null = obs_null/exp_null
            if value is None:
                idx = np.unravel_index(np.argmax((factor - 1).abs().to_numpy()), factor.shape)
                results[k] = factor[idx[1]].to_dict()
                results['{}_null'.format(k)] = factor_null[idx[1]].to_dict()
            else:
                results[k] = factor[value[k]].to_dict()
                results['{}_null'.format(k)] = factor_null[value[k]].to_dict()
        elif dtype=='continuous':
            results[k] = sub_res['log2_grpVall']
            results['{}_null'.format(k)] = sub_res_null['log2_grpVall']
        else:
            print('wrong dtype specified in voi')
    return results

def get_max_value(df, 
                  voi={'ptb_37wks': 'categorical','GA': 'continuous'},
                  groupby='Error group'):
    value = {}
    for i, (k, dtype) in enumerate(voi.items()):
        sub_res = md_group_diffs(df, {k:voi[k]}, groupby=groupby)[k]
        if dtype=='categorical':
            # obs
            obs = pd.DataFrame(sub_res['pObserved'])
            exp = pd.DataFrame(sub_res['pExpected'])
            factor = obs/exp

            idx = np.unravel_index(np.argmax((factor - 1).abs().to_numpy()), factor.shape)
            value[k] = factor[idx[1]].name
        else:
            continue
    return value