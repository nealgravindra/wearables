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