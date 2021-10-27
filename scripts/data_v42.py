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
        
        self.stop()
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
        for problem in self.exclude_list:
            if isinstance(self.exclude[problem], list):
                for k in list(self.exclude[problem]):
                    if verbose:
                        print('{} excluded because {}'.format(k, problem))
                    counter += 1
                    del self.IDs[k]
                    del self.data[k]
            else:
                if verbose:
                    print('{} excluded because {}'.format(self.exclude[problem], problem))
                counter += 1
                del self.data[self.exclude[problem]]
                del self.IDs[self.exclude[problem]]
            if verbose:
                print('{} measurements deleted'.format(counter))
            
    def drop_IDs_in_md_notin_data(self, del_missing_labels):
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

                dt = raw_md[k].value_counts(normalize=True, dropna=False).reset_index()
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
            flag = f.split('binarize_is_')[1]
            x_i = (x_i == flag).astype(int)
        elif 'binarize_not' in f:
            flag = f.split('binarize_is_')[1]
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

        for i, (k, v) in enumerate(voi.items()):
            x_i = md[k]
            if v[0] is not None:
                for f in v[0]:
                    x_i = md_filters(x_i, f)

            # type to save
            if v[1] == 'categorical':
                x_i = x_i.astype(str)
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
    
    def add_labels_to_data():
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
    outs = {'data': rawdata.data, 'md_summary': rawdata.md_summary, 'voi': rawdata.voi}
    with open(filename, 'wb') as f:
        pickle.dump(outs, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    print('Data written to {}'.format(filename)) 
    return rawdata
            
            
            
            
            
            
            
            

def pad_align_transform(data):
    X = np.empty((len(data.keys()), 24*60))
    y = np.empty((len(data.keys()),))
    for i, (k, v) in enumerate(data.items()):
        ts, act = v[0], v[1]
        first_hour, first_min = ts[0].hour, ts[0].minute
        zeros2pad = np.zeros((first_hour*60 + first_min + 1))
        act = np.concatenate((zeros2pad, [float(i) for i in act[:24*60 - zeros2pad.shape[0]]]))
        if act.shape[0] < 24*60:
            act = np.concatenate((act, np.zeros((24*60-act.shape[0], ))))

        # add log-pseudocount
        act = np.log(act + 1)
        X[i, :] = act
        y[i] = int(k.split('-')[1])
    return X, y

def get_train_test():
    data = load_pp_actigraphy()
    data_train, data_test = split_pp_actigraphy(data)
    X_train, y_train = pad_align_transform(data_train)
    X_test, y_test = pad_align_transform(data_test)
    return {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}

def load_actigraphy_metadata(fname='/home/ngr/gdrive/wearables/data/MOD_1000_Woman_Activity_Data/MOD_Data_2021.csv'):
    return pd.read_csv(fname, index_col=0)


def pad_align_transform_yID(data):
    '''Treat y as identifier to map with dictionary version of metadata outcomes

    '''
    X = np.empty((len(data.keys()), 24*60))
    y = pd.DataFrame(columns=['pid', 'GA'], index=list(range(len(data.keys()))))
    for i, (k, v) in enumerate(data.items()):
        ts, act = v[0], v[1]
        first_hour, first_min = ts[0].hour, ts[0].minute
        zeros2pad = np.zeros((first_hour*60 + first_min + 1))
        act = np.concatenate((zeros2pad, [float(i) for i in act[:24*60 - zeros2pad.shape[0]]]))
        if act.shape[0] < 24*60:
            act = np.concatenate((act, np.zeros((24*60-act.shape[0], ))))

        # add log-pseudocount
        act = np.log(act + 1)
        X[i, :] = act
        y.loc[i, 'pid'], y.loc[i, 'GA'] = k.split('-')[0], int(k.split('-')[1])
    return X, y


def get_train_test_yID():
    data = load_pp_actigraphy()
    data_train, data_test = split_pp_actigraphy(data)
    X_train, y_train = pad_align_transform_yID(data_train)
    X_test, y_test = pad_align_transform_yID(data_test)
    return {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}


# @metadata


def load_data_md(dfp, onehot_md=False):
    '''Load filtered cohort and metadata columns by PID for modeling.

    NOTE:
      - the two files indicated are the minimal package for data transfer
    '''
    datapkl_file=os.path.join(dfp, 'ppdata_1wk.pkl')
    md_file = os.path.join(dfp, 'MOD_Data_2021.csv')
    data = load_ppdata(filepath=datapkl_file)
    
    # load metadata
    md = load_rawmd(filepath=md_file)
    ppdata, md = md_data_keymatch(data, md)
    
    ppmd = pp_metadata(md, voi)
    
    if onehot_md:
        raise NotImplementedError 
    return ppdata, ppmd

# train/test data with user specified label
def split_by_pid(X, df, prop_train=0.8):
    '''Split data by patient ID.
    '''
    pids = np.unique(df['pid'].to_list())
    train_pids = np.random.choice(pids, int(len(pids)*prop_train), replace=False)
    test_pids = [i for i in pids if i not in train_pids]
    train_idx = np.where(df['pid'].isin(train_pids))[0] #df.loc[df['pid'].isin(train_pids), :].index.to_list()
    X_train, y_train = X[train_idx, :], df.iloc[train_idx, :]
    val_pids = np.random.choice(test_pids, int(len(test_pids)*0.5), replace=False)
    val_idx = np.where(df['pid'].isin(val_pids))[0]
    X_val, y_val = X[val_idx, :], df.iloc[val_idx, :]
    test_pids = [i for i in test_pids if i not in val_pids]
    test_idx = np.where(df['pid'].isin(test_pids))[0] 
    X_test, y_test = X[test_idx, :], df.iloc[test_idx, :]
    return X_train, y_train, X_val, y_val, X_test, y_test

# data transforms
def logpseudocount(X):
    return np.log(X + 1.)

def standardize(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def get_Xy(data, md, pkl_out=None, prop_train=0.8, transform=True, single_precision=True, verbose=False):
    '''Get actigraphy data and all possible labels for model dev

    TODO:
      - [ ] (enhancement): label encoding before returning data
    '''
    X = np.zeros((len(data.keys()), data[list(data.keys())[0]][1].shape[0]))
    label_df = pd.DataFrame(index=np.arange(len(data.keys())))
    for i, (k, v) in enumerate(data.items()):
        X[i, :] = v[1] # v[0] has datetimes
        pid = k.split('-')[0]
        label_df.loc[i, 'pid'] = pid
        label_df.loc[i, 'GA'] = float(k.split('-')[1])
    md['record_id'] = md['record_id'].astype(str)
    label_df = label_df.merge(md, left_on='pid', right_on='record_id', how='left')
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_pid(X, label_df, prop_train=prop_train)
    if transform:
        X_train = logpseudocount(X_train)
        X_val = logpseudocount(X_val)
        X_test = logpseudocount(X_test)
    if verbose:
        # print shapes
        print('X_train: ({}, {})'.format(*X_train.shape))
        print('y_train: ({}, {})'.format(*y_train.shape))
        print('X_val: ({}, {})'.format(*X_val.shape))
        print('y_val: ({}, {})'.format(*y_val.shape))
        print('X_test: ({}, {})'.format(*X_test.shape))
        print('y_test: ({}, {})'.format(*y_test.shape))
        if single_precision:
            X_train = X_train.astype('float32')
            X_val = X_val.astype('float32')
            X_test = X_test.astype('float32')
    out = {'X_train': X_train, 'Y_train': y_train, 
           'X_val': X_val, 'Y_val': y_val, 
           'X_test': X_test, 'Y_test': y_test}
    if pkl_out is not None:
        with open(pkl_out, 'wb') as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
   
    return out

class actigraphy_dataset(torch.utils.data.Dataset):
    def __init__(self, X, y_wide):
        self.X = X
        if y_wide.shape[0] != y_wide.size:
            y_wide = self.wide2long(y_wide) # one-hot encoded to long
        self.y = y_wide

    def wide2long(self, y):
        return np.argmax(y, 1)

    def __len__(self):
        assert self.X.shape[0] == self.y.shape[0], 'Data indices do NOT align'
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx, :].reshape(1, -1))
        y = torch.tensor(self.y.iloc[idx])
        X = X.float()
        y = y.float()
        return {'X':X,
                'y':y,
                'idx':idx}

def pkl_loadedactigraphy_md(dfp='/home/ngr4/project/wearables/data/processed/', pkl_out=None, onehot_md=True, minority_as_pos=True):
    total = time.time()
    data, md = load_data_md(dfp)
    print('time elapsed: {:.0f}-s'.format(time.time() - total))
    if onehot_md:
        ohmd = pd.get_dummies(md, dtype=np.int64)
        if minority_as_pos:
            for c in ohmd.columns:
                if ohmd[c].dtype != 'float32':
                    if not ohmd[c].sum()/ohmd.shape[0] >= 0.5:
                        new_c = ''.join(i+'_' for i in c.split('_')[:-1]) + 'not'+ c.split('_')[-1]
                        ohmd[new_c] = (~(ohmd[c]==1)).astype(np.int64)
                        del ohmd[c]
        out = {'data': data, 'ohmd': ohmd, 'md': md}
    else:
        out = {'data': data, 'md': md}

    if pkl_out is not None:
        with open(os.path.join(dfp, pkl_out), 'wb') as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
    return out

def load_actigraphy_md(dfp='/home/ngr4/project/wearables/data/processed', pkl='data_md_210929.pkl'):
    with open(os.path.join(dfp, pkl), 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data

if __name__ == '__main__':
    # data = weardata.pkl_loadedactigraphy_md(pkl_out='data_md_210929.pkl')

    # save data for the first time 
    data = pkl_loadedactigraphy_md(pkl_out='data_md_210929.pkl')
    data = load_actigraphy_md()
    data, md, _ = data['data'], data['ohmd'], data['md']
    model_data = get_Xy(data, md, 
                        pkl_out='/home/ngr4/project/wearables/data/processed/model_data_210929.pkl')

    ####
    # SPECIFY
    ####
    new_split = False # if want to use same model data, load old; else, start from scratch
    ####

    if new_split:
        data = load_actigraphy_md()
        data, md, _ = data['data'], data['ohmd'], data['md']
        model_data = get_Xy(data, md) # save exp_trial details here, into temp; pull only top-1 to long-term storage
    else:
        model_data = load_datadict(fname='/home/ngr4/project/wearables/data/processed/model_data_210929.pkl')

    # select target
    X_train, y_train = model_data['X_train'], model_data['Y_train'].loc[:, target]
    X_val, y_val = model_data['X_val'], model_data['Y_val'].loc[:, target]
    X_train, y_train = model_data['X_test'], model_data['Y_test'].loc[:, target]


