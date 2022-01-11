import pickle
import numpy as np
import pandas as pd
import os
import glob
import warnings
import time
import re
import datetime
import pyActigraphy

import sys
sfp = '/home/ngrav/project' 
sys.path.append(sfp)
from wearables.scripts import utils as wearutils

import torch

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', size = 9)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
plt.rcParams['legend.frameon']=False
plt.rcParams['axes.grid']=False
plt.rcParams['legend.markerscale']=0.5
plt.rcParams['savefig.dpi']=600
sns.set_style("ticks")

class raw2df():
    ''' 
    Description:
      The data was sent with {PID}_GA{y}.mtn names. We need to convert this
        raw data into numpy and pandas. First, we identify all valid files and
        give new modeling IDs, {PID}_{y} where y is GA in wks. 
        
    Assumptions:
      all files should be in a folder, where the filenames indicate PID_GA.mtn, 
        such that the raw data is in raw_fp+'*/*.mtn'
    '''
    def __init__(self,
      md_filter_dict=None,
      nb_days=7, frompkl=None,
      log_pseudocount=True,
      raw_fp='/home/ngrav/project/wearables/data/raw/MOD1000WomanActivityData20210707T213505Z-001/MOD 1000 Woman Activity Data/', 
      raw_md_fp='/home/ngrav/project/wearables/data/raw/MOD_Data_2021.csv'):
        
        # initialize, load md
        self.timer = wearutils.timer()
        self.timer.start()
        self.raw_fp = raw_fp
        self.raw_md = pd.read_csv(raw_md_fp, low_memory=False)
        self.exclude = {'no_lux': [], 'chk_t': [], 
                        'lt_1d': [], 'lt_max_t': [],
                        'corrupt_mtn': [], 
                        'md_mismatch': []} 
        self.nb_days = nb_days
        self.data = dict()
        
        # get "{PID}_{GA}" from filenames
        self.IDs = self.ids_from_filenames(verbose=False)
        self.timer.stop() # done with initializations
        
        # load all
        self.timer.start()
        self.load_all()
        self.timer.stop()
        print('\nRaw actigraphy data loaded in {:.1f}-min'.format(self.timer.sum()/60))
        
        # get valid IDs
        self.timer.start()
        self.delete_problematic_measurements()
        
        # pre-process metadata
        self.drop_IDs_in_md_notin_data()
        
        # import voi
        from wearables.data.processed import md_pp_specification as md_rules
        self.voi = md_rules.voi
        md_summary = self.inspect_tabular(self.raw_md, self.voi, plot_out='/home/ngrav/project/wearables/results/md_var_summary_pre.pdf')
        self.md = self.pp_metadata(self.raw_md, self.voi)
        self.md_summary = self.inspect_tabular(self.md, self.voi, plot_out='/home/ngrav/project/wearables/results/md_var_summary_post.pdf')
        
        # add metadata as labels to each data file
        self.add_labels_to_data()
        
        # transforms
        if log_pseudocount:
            self.transform_logpseudocount()
        
        self.timer.stop()
        print('\nAll raw2modeldata pp done in {:.1f}-min'.format(self.timer.sum()/60))
            
    def ids_from_filenames(self, verbose=False):
        def GA_from_md(filename, md, verbose=verbose):
            '''Grab the GA from the csv file OR assume that T1 corresponds
                 to `gawks_enroll`
            '''
            f = os.path.split(filename)[1]
            newID = re.search(r'(\d*)(.*).mtn', f, re.I)
            pid = int(newID.group(1))
            if len(str(pid)) != 4 and '1T' in f:
                newID = re.search(r'(\d*)1T(.*).mtn', f, re.I)
                pid = int(newID.group(1))
            GA = md.loc[md['record_id']== pid, 'gawks_enroll'].item()
            return str(pid), str(GA)
        IDs = dict()
        files = glob.glob(os.path.join(self.raw_fp, '*/*.mtn'))
        for f in files:
            fname = os.path.split(f)[1]
            ID = re.search(r'(.\d*)(.*)GA(.\d*).mtn', fname, re.I)
            if ID is None:
                pid, GA = GA_from_md(f, self.raw_md)
            else:
                pid = ID.group(1) if 'P' not in fname else ID.group(2).split('_')[1] # exception for erroneous entry
                GA = ID.group(3)
            IDs['{}_{}'.format(pid, GA)] = f

        if verbose:
            # chk problematic ones
            for i in IDs.keys():
                if i.split('_')[0] in ['2014', '2003', '1406', '1647'] or len(i.split('_')[0]) != 4:
                    print(i, os.path.split(IDs[i])[1])
                    print('')
        return IDs
    
    def pyactigraphy_read_raw(self, ID, file):
        try:
            raw = pyActigraphy.io.read_raw_mtn(file)
        except AttributeError:
            self.exclude['corrupt_mtn'].append(ID)
            print('Corrupt file for ID: {}'.format(ID))
            return None
        if raw.light is None:
            self.exclude['no_lux'].append(ID)
            light = []
        else:
            light = raw.light
        t = raw.data.index.to_series()
        sleep = raw.Oakley(threshold=80) # 
        activity = raw.data
        del raw
        
        # get start
        first_midnight_idx = [i for i, t_i in enumerate(t) if t_i.hour==0 and t_i.minute==0]
        if len(first_midnight_idx) != 0:
            first_midnight_idx = first_midnight_idx[0] 
        else:
            first_midnight_idx = 0
            self.exclude['lt_1d'].append(ID)
            
        # go from first midnight to nb-days out, fill with NA otherwise
        new_t = pd.date_range(t.iloc[first_midnight_idx], 
                              t.iloc[first_midnight_idx] + datetime.timedelta(days=self.nb_days), 
                              periods=self.nb_days*24*60+1)
        t = t.reindex(new_t)
        activity = activity.reindex(new_t)
        sleep = sleep.reindex(new_t)
        light = light.reindex(new_t) if isinstance(light, pd.Series) else []
        
        # log missingness to chk tdelta 
        if any(t.isna()):
            self.exclude['chk_t'].append(ID)

        return {'t':t, 'activity':activity, 'light':light, 'sleep':sleep}
    
    def load_all(self):
        for i, k in enumerate(self.IDs.keys()):
            self.data[k] = self.pyactigraphy_read_raw(k, self.IDs[k])
            
    def delete_problematic_measurements(self, 
                                        exclude_list=['no_lux', 'corrupt_mtn', 'lt_1d', 'chk_t'], 
                                        verbose=False):
        counter = 0 
        for problem in exclude_list:
            if isinstance(self.exclude[problem], list):
                for k in list(self.exclude[problem]):
                    if verbose:
                        print('{} excluded because {}'.format(k, problem))
                    counter += 1
                    try:
                        del self.IDs[k]
                    except KeyError: # already deleted, move to nxt key
                        continue
                    del self.data[k]
            else:
                if verbose:
                    print('{} excluded because {}'.format(self.exclude[problem], problem))
                counter += 1
                try:
                    del self.data[self.exclude[problem]] 
                except KeyError: # already deleted, move on to nxt problem
                    continue
                del self.IDs[self.exclude[problem]]
            if verbose:
                print('{} measurements deleted'.format(counter))
            
    def drop_IDs_in_md_notin_data(self, del_missing_labels=False):
        IDs = list(self.IDs.keys())
        self.exclude['missing_md'] = [i for i in IDs if int(i.split('_')[0]) not in self.raw_md['record_id'].tolist()]
        if len(self.exclude['missing_md']) != 0:
            warnings.warn('No label!')
            print('problematic keys that have no labels:')
            counter = 0
            for k in self.exclude['missing_md']:
                print(k)
                if del_missing_labels:
                    del self.IDs[k]
                    del self.data[k]
                    counter += 1
            if del_missing_labels:
                print('{} measurements deleted.'.format(counter))
                
        # remove from md
        self.raw_md = self.raw_md.loc[self.raw_md['record_id'].isin([int(i.split('_')[0]) for i in IDs] ), :]
        
    def compute_composities_in_md(self):
        raise NotImplementedError
        
    def inspect_tabular(self, tab, voi, figsize=(36, 24), n_subplots=[11, 10], verbose=False, plot_out=None, summary_out=None):
        '''

        Arguments:
          voi (dict): variables of interest as dictionary with 
            format: key=var name in column of df,
            value=([list of filters], continuous/categorical/ordinal) where 
            list of filters are triggered by a separate function later, and the second
            element of the tuple is a specification for the var type to be transformed.

        '''
        fig = plt.figure(figsize=figsize)
        summary = pd.DataFrame()
        from scipy.stats import normaltest
        for i, (k, v) in enumerate(voi.items()):
            ax = fig.add_subplot(n_subplots[1], n_subplots[0], i+1)
            if v[1] == 'continuous':
                statistic, p = normaltest(tab[k])
                if p < 0.01:
                    central_summary = '{:.2f}'.format(tab[k].median())
                    spread = ' ({:.2f} - {:.2f})'.format(tab[k].quantile(0.25), tab[k].quantile(0.75))
                else:
                    central_summary = '{:.2f}'.format(tab[k].mean())
                    spread = ' ({:.2f})'.format(tab[k].std())

                # calculate outliers
                IQR = tab[k].quantile(0.75) - tab[k].quantile(0.25)
                lo_bar = tab[k].median() - 1.5*IQR
                hi_bar = tab[k].median() + 1.5*IQR

                value_props = tab[k].loc[(tab[k] > hi_bar) | (tab[k] < lo_bar)].to_list() # could save fit kernle from distplot?


                if verbose:
                    print(k, ':')
                    print('  p_nan: {:.2f}'.format(tab[k].isna().sum()/tab.shape[0]))
                    print('  outliers: ', np.unique(value_props))

                # add distplot
                sns.distplot(tab[k], label=central_summary+spread, ax=ax)
                ax.legend()
                ax.set_xlabel('') # replace with title
                ax.set_title(k)

            elif v[1] == 'categorical':

                dt = tab[k].value_counts(normalize=True, dropna=False).reset_index()
                dt['index'] = dt['index'].astype(str) # force nan to show up on plot
                value_props = dt.to_dict()

                if verbose:
                    print(k, ':')
                    print('  p_nan: {:.2f}'.format(tab[k].isna().sum()/tab.shape[0]))
                    print('  value_props: ', np.unique(value_props))

                # add barplot 
                sns.barplot(x='index', y=k, data=dt, ax=ax)
                ax.set_ylabel('Proportion')
                ax.set_xlabel('')
                ax.set_title(k)

            else:
                raise NotImplementedError
                print('Variable could not be processed as instructions for {} not implemented.'.format(v[1]))

            fig.tight_layout()
            # store data
            dtt = pd.DataFrame({'Variable':k, 
                                'Type':v[1],
                                'value_props_OR_outliers':None, 
                                'p_nan':tab[k].isna().sum()/tab.shape[0]}, 
                               index=[0])
            dtt.at[0, 'value_props_OR_outliers'] = value_props
            summary = summary.append(dtt, ignore_index=True)

            if plot_out is not None:
                fig.savefig(plot_out, dpi=600, bbox_inches='tight')

            if summary_out is not None:
                summary.to_csv(summary_out)

        return summary
        
    def md_filters(self, x_i, f, verbose=True):
        # filters
        def nan2n(x, n=7):
            return x.replace([np.nan], int(n))

        def mean_impute(x):
            return x_i.fillna(x_i.mean()) # nanmean

        def n2nan(x, n=-99):
            return x.replace([int(n)], np.nan)

        def n2n(x, n1=2, n2=0):
            return x.replace(n1, n2)

        def absval(x):
            return x.abs()

        # flow
        if 'nan2' in f:
            x_i = nan2n(x_i, n=f.split('nan2')[1])
        elif '2nan' in f:
            x_i = n2nan(x_i, n=f.split('2nan')[0])
        elif f == 'mean_impute': # technically erroneous because need train/test split info
            x_i = mean_impute(x_i)
        elif 'to' in f:
            x_i = n2n(x_i, n1=f.split('to')[0], n2=f.split('to')[1])
        elif f == 'absval':
            x_i = absval(x_i)
        elif 'binarize_is' in f:
            flag = float(f.split('binarize_is_')[1])
            x_i = (x_i == flag).astype(int)
        elif 'binarize_not' in f:
            flag = float(f.split('binarize_not_')[1])
            x_i = (x_i != flag).astype(int)
        else:
            warnings.warn('Warning. Transform not recognized')
            if verbose:
                print('  \nTransformation for {} variable skipped.\n'.format(x_i.name))
        return x_i


    def pp_metadata(self, md, voi, out_file=None):
        '''Pre-process metadata csv.

        Arguments:
          md (pd.DataFrame): metadata read in with record_id as pid from csv file
          voi (dict): keys are variable name in metadata and values are tuples, specifying
            (transform, dtype) where transform is a string for a function and dtype is categorical
            or continuous. All continuous var will be stored as np.float32, and categorical is a flag
            to later one-hot-encode (non-ordinal numbers can still be stored). Transform can have
        '''
        ppmd = pd.DataFrame()
        ppmd['record_id'] = md['record_id']
        self.cat_class_encoding = dict()
        
        for i, (k, v) in enumerate(voi.items()):
            x_i = md[k]
            if v[0] is not None:
                for f in v[0]:
                    x_i = self.md_filters(x_i, f)

            # type to save
            if v[1] == 'categorical':
                x_i = x_i.astype(str)
                self.cat_class_encoding[k] = {value:float(i) for i, value in enumerate(np.sort(x_i.unique()))}
                x_i = x_i.map(self.cat_class_encoding[k])
            elif v[1] == 'continuous':
                x_i = x_i.astype(np.float32) # single precision
            else:
                warnings.warn('Warning. Data type not recognized')
                print('\nData type for {} left as default type\n'.format(k))
            ppmd[k] = x_i

        if out_file is not None:
            metadata = {}
            metadata['md'] = ppmd
            metadata['variables'] = voi
            with open(out_file, 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()
        return ppmd
    
    def add_labels_to_data(self):
        for k in self.data.keys():
            pid = int(k.split('_')[0])
            dt = self.md.loc[self.md['record_id']==pid, :]
            if dt.shape[0] != 1:
                warnings.warn('Problem with metadata')
                self.exclude['md_mismatch'].append(k)
                print('Check {} in self.data'.format(k))
            labels = {col: dt.iloc[0, j] for j, col in enumerate(dt.columns)} 
            self.data[k]['md'] = labels
            
    def transform_logpseudocount(self, data_keys=['light', 'activity']):
        '''Transform light and activity
        '''
        def logpseudocount(x):
            return np.log(x + 1.0)
        
        # loop through all data
        for k in self.data.keys():
            for data_cat in data_keys:
                self.data[k][data_cat] = logpseudocount(self.data[k][data_cat])        

def save_raw2df(filename):
    rawdata = raw2df()
    outs = {'data': rawdata.data, 'md_summary': rawdata.md_summary, 'voi': rawdata.voi, 'cat_class_enc':rawdata.cat_class_encoding}
    with open(filename, 'wb') as f:
        pickle.dump(outs, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    print('Data written to {}'.format(filename)) 
    return rawdata

class actigraphy(torch.utils.data.Dataset):
    def __init__(self, ids, data, target_name):
        self.ids = ids
        self.data = data
        self.target_name = target_name
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        k = self.ids[idx]
        x = torch.cat((torch.tensor(self.data['data'][k]['activity'], dtype=torch.float32)[:-1].reshape(1, -1), 
            torch.tensor(self.data['data'][k]['light'], dtype=torch.float32)[:-1].reshape(1, -1)), dim=0)
        if self.target_name == 'GA':
            try:
                y = torch.tensor(float(k.split('_')[-1]), dtype=torch.float32)
            except ValueError:
                print(k)
        else:
            y = torch.tensor(float(self.data['data'][k]['md'][self.target_name]), dtype=torch.float32)
        return {'x': x, 'y': y, 'id': k}
    
    
class torch_dataloaders():
    def __init__(self, target_name, prop_trainset=1., batch_size=32, filename='/home/ngrav/data/wearables/processed/MOD1000_modeldata.pkl'):
        self.data = self.load_preproced(filename)
        self.target_name = target_name
        self.batch_size = batch_size
        self.tasktype = 'regression' if target_name=='GA' else self.data['voi'][target_name][1]
        self.prop_trainset = prop_trainset
        
        # split data and get dataloaders
        self.split_data()
        self.get_dataloaders()
        
    def load_preproced(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            f.close()
        return data
    
    def split_data(self, train_ratio=0.7):
        pids = np.unique([i.split('_')[0] for i in self.data['IDs'].keys()])
        train_pids = np.random.choice(pids, int(len(pids)*train_ratio*self.prop_trainset), replace=False)
        test_pids = [i for i in pids if i not in train_pids]
        val_pids = np.random.choice(test_pids, int(len(test_pids)*0.5), replace=False)
        
        # to modeling IDs
        self.train_ids = [i for i in self.data['IDs'].keys() if i.split('_')[0] in train_pids]
        self.val_ids = [i for i in self.data['IDs'].keys() if i.split('_')[0] in val_pids]
        self.test_ids = [i for i in self.data['IDs'].keys() if i.split('_')[0] in test_pids]

    def get_dataloaders(self):
        self.train_dl = torch.utils.data.DataLoader(
            actigraphy(self.train_ids, self.data, self.target_name),
            batch_size=self.batch_size,
            num_workers=12,
            shuffle=True, pin_memory=True)
        self.val_dl = torch.utils.data.DataLoader(
            actigraphy(self.val_ids, self.data, self.target_name),
            batch_size=self.batch_size,
            num_workers=12,
            shuffle=True, pin_memory=True)
        self.test_dl = torch.utils.data.DataLoader(
            actigraphy(self.test_ids, self.data, self.target_name),
            batch_size=self.batch_size,
            num_workers=12,
            shuffle=True, pin_memory=True)

class dataloader():
    '''Load data into numpy data types and specify number of folds. If kfold=1 or 0, then single val set returned.
    '''
    def __init__(self, target_name='GA', kfold=5, prop_trainset=1., include_lux=False, filename='/home/ngrav/data/wearables/processed/MOD1000_modeldata.pkl'):
        self.rawdata_file = filename
        self.data = self.load_preproced(filename)
        self.target_name = target_name
        self.tasktype = 'regression' if target_name=='GA' else self.data['voi'][target_name][1]
        self.kfold = kfold
        self.prop_trainset = prop_trainset
        self.include_lux = include_lux
        
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
                y[i] = np.float32(self.data['data'][k]['md'][target_name])
        # define in terms of category class
        try:
            enc = self.data['cat_class_enc'][target_name]
        except KeyError:
            enc = None
        if enc is not None and len(enc.keys()) > 2:
            # force to long 
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






