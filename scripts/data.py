import pickle
import numpy as np
import pandas as pd
import os
import glob
import warnings
import time

import sys
sys.path.append('/home/ngr/gdrive/wearables/scripts')
import utils as wearutils

def pkldata(filepath='/home/ngr/gdrive/wearables/data/raw/MOD 1000 Woman Activity Data-20210707T213505Z-001/MOD 1000 Woman Activity Data',
            pkl_out=None):

    # copy over file if it does not exist, otherwise skip poor organization
    for file_in_outer_folders in glob.glob(os.path.join(filepath, '*csv')):
        pid = os.path.split(file_in_outer_folders)[1].split('_')[0]

        pid_folder_count = 0
        pid_folder = []
        for i in glob.glob(os.path.join(filepath, '*'+pid+'*')):
             if os.path.isdir(i):
                 pid_folder_count += 1
                 pid_folder.append(i)
        if pid_folder_count == 1:
            dst = os.path.join(pid_folder[0], os.path.split(file_in_outer_folders)[1])
            if not os.path.exists(dst):
                from shutil import copyfile
                print('  copying from outer folder to:\n    {}'.format(dst))
                copyfile(file_in_outer_folders, dst)
        elif pid_folder_count == 0:
            os.mkdir(os.path.join(filepath, pid))
            from shutil import copyfile
            dst = os.path.join(os.path.join(filepath, pid), os.path.split(file_in_outer_folders)[1])
            print('  copying from outer folder to:\n    {}'.format(dst))
            copyfile(file_in_outer_folders, dst)
        else:
            warnings.warn('Warning.\n  file in outer folder not associated with PID:\n    {}'.format(file_in_outer_folders))

    # read and store with pid_GA label
    print('\nLoading graphs into datapkl...')
    data = {}
    counter = 0
    noga_counter = 0
    noga_files = []
    tic = time.time()
    for folder in glob.glob(os.path.join(filepath, '*')):
        for file in glob.glob(os.path.join(folder, '*.csv')):
            if 'ga' not in file.lower():
                # skip these for now, until resolved
                noga_files.append(file)
                noga_counter += 1
            else:
                dt = pd.read_csv(file)
                f = os.path.split(file)[1]
                f = f.replace(' ', '')
                if '_' in f:
                    pid, f = f.lower().split('_ga')
                elif '-' in f:
                    pid, f = f.lower().split('-ga')
                GA = int(f.split('.csv')[0])

                for i, row0 in enumerate(dt.iloc[:, 0]):
                    if isinstance(row0, str):
                        if np.sum([True if '/' in ii else False for ii in row0]) == 2:
                            row_idx_start = i
                            break

                idx = dt.iloc[row_idx_start:, 0].loc[(~dt.iloc[row_idx_start:, [0,1,2]].isna().any(1)) == True].index.to_list()
                t = pd.to_datetime(dt.iloc[idx, 0].astype(str) + ' ' + dt.iloc[idx, 1].astype(str), format='%m/%d/%Y %I:%M:%S %p',)
                activity = dt.iloc[idx, 2] # MW counts
                data['{}-{}'.format(pid, GA)] = [t.to_list(), activity.to_list()]
                counter += 1

                if counter % 100 == 0 :
                    print('... through {} graphs in {:.0f}-s'.format(counter, time.time() - tic))

    print('\n... Finished! {} graphs loaded in {:.1f}-min'.format(counter, (time.time() - tic)/60))

    print('\  NOTE: the following {} samples were not processed (no GA label):'.format(noga_counter))
    for f in noga_files:
        print('    ', f)

    if pkl_out is not None:
        if not os.path.exists(os.path.split(pkl_out)[0]):
            os.mkdir(os.path.split(pkl_out)[0])
        with open(pkl_out, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        print('Wrote unprocessed data to {}'.format(pkl_out))
    return data

def load_datadict(fname='/home/ngr/gdrive/wearables/data/processed/GAactigraphy_datadict.pkl'):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data

def keys2_npt_nobs(keys_list):
    PID, GA = [], []
    for k in keys_list:
        pid, ga = k.split('-')
        PID.append(pid)
        GA.append(int(ga))
    return np.unique(PID).shape[0], len(GA)

def chk_tdeltas(data, truncate_at_day=None, leading_pad=True, drop_first_day=True, verbose=True):
    '''Check for the temporal differences between consecutive measurements.

    Arguments:
      truncate_at_day (int): (optional, Default=None) in units of d, truncate after x days. None keeps all data
      leading_pad (bool): (optional, Default=True) add pads to align ts so first time is 12 AM
      drop_first_day (bool): (optional, Default=True) drop the first day since, worst case is measurement
        starting at 12 AM
    '''

    k_irreg_tdelta = []
    irreg_tdelta = []
    keys2keep = []
    k_lt1d = []
    timer = wearutils.timer()

    if verbose:
        timer.start()
    for i, (k, v) in enumerate(data.items()):
        ts, _ = v[0], v[1]
        if leading_pad:
            ts = pd.to_datetime(np.linspace(ts[0].replace(hour=0, minute=0).value, ts[0].value, ts[0].hour*60 + ts[0].minute + 1)).to_list() + ts
        if drop_first_day:
            # drop 1d* 24h/d * 60min/h timepts
            idx = 1*24*60+1
            if not len(ts) <= idx:
                ts = ts[idx:]
            else:
                k_lt1d.append(k)
        if truncate_at_day is not None:
            ts = ts[:truncate_at_day*24*60+1]
        unique_tdeltas = np.unique(np.diff(pd.Series(ts)))
        if np.timedelta64(1, 'm') in unique_tdeltas and unique_tdeltas.shape[0] == 1:
            keys2keep.append(k)
        else:
            k_irreg_tdelta.append(k)
            irreg_tdelta.append(unique_tdeltas)
        if verbose:
            if i % 500 == 0 and i != 0 :
                print('  through m={}\t{:.0f}-s elapsed'.format(i+1, timer.stop()))

    # filter out keys of data with less than 1d
    keys2keep =  [i for i in keys2keep if i not in k_lt1d]

    if verbose:
        print('\n{} measurements (out of {}) have multiple t delta'.format(len(k_irreg_tdelta), len(data.keys())))
        npt, nobs = keys2_npt_nobs(keys2keep)
        print('  i.e., {}-pts across {}-measurements have 1min t deltas, whereas'.format(npt, nobs))
        npt, nobs = keys2_npt_nobs(k_irreg_tdelta)
        print('        {}-pts across {}-measurements have more than 1 t delta'.format(npt, nobs))

        print('\n  {} measurements had fewer than 1d data. Were not added to keys2keep'.format(len(k_lt1d)))

    if verbose:
        print('  \nunique time deltas:')
        print(pd.value_counts([j for i in irreg_tdelta for j in i]))

    return keys2keep, k_irreg_tdelta

def chk_tdeltas_multiday(data, days_to_truncate=[None, 3, 5, 7, 14], out_file=None, verbose=True):
    results = pd.DataFrame()
    if verbose:
        timer = wearutils.timer()
    for d in days_to_truncate:
        if verbose:
            timer.start()
        keys2keep, keys2discard = chk_tdeltas(data, truncate_at_day=d, verbose=False)
        npt_keep, nobs_keep = keys2_npt_nobs(keys2keep)
        npt_discard, nobs_discard = keys2_npt_nobs(keys2discard)
        results = results.append(pd.DataFrame({'truncate_after_day':d,
                                               'npt2keep':npt_keep,
                                               'nmeas2keep':nobs_keep,
                                               'npt2discard':npt_discard,
                                               'nmeas2discard':nobs_discard,}, index=[0]), ignore_index=True)
        if verbose:
            print('\tthrough {}\tin {:.0f}-s\tt_elapsed: {:.0f}-min'.format(str(d), timer.stop(), timer.sum()/60))
    if out_file is not None:
        results.to_csv(out_file)
    return results


def get_ntimepoints(data, truncate_at_day=None, leading_pad=True, drop_first_day=False, verbose=True):
    '''Get n_timepoints after adding leading zeros and, optionally, dropping first day

    Arguments:
      truncate_at_day (int): (optional, Default=None) in units of d, truncate after x days. None keeps all data
      leading_pad (bool): (optional, Default=True) add pads to align ts so first time is 12 AM
      drop_first_day (bool): (optional, Default=True) drop the first day since, worst case is measurement
        starting at 12 AM
    '''
    n_t = pd.DataFrame()
    k_lt1d = []

    for i, (k, v) in enumerate(data.items()):
        pid, ga = k.split('-')
        ts, _ = v[0], v[1]
        if leading_pad:
            ts = pd.to_datetime(np.linspace(ts[0].replace(hour=0, minute=0).value, ts[0].value, ts[0].hour*60 + ts[0].minute + 1)).to_list() + ts
        if drop_first_day:
            # drop 1d* 24h/d * 60min/h timepts
            idx = 1*24*60+1
            if not len(ts) <= idx:
                ts = ts[idx:]
            else:
                k_lt1d.append(k)
        if truncate_at_day is not None:
            ts = ts[:truncate_at_day*24*60+1]
        n_t = n_t.append(pd.DataFrame({'key':k, 'pid':pid, 'GA':ga, 'n_t':len(ts), }, index=[0]), ignore_index=True)
    n_t['n_days'] = n_t['n_t']/(24*60)
    return n_t

def viz_prop_n_days(data, day_cutoff=8, out_file=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # get data
    nt = get_ntimepoints(data)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    sns.ecdfplot(x='n_days', data=nt, ax=ax, label='All data')
    sns.ecdfplot(x='n_days', data=nt.groupby('GA').mean().reset_index(), ax=ax, label='GA')
    sns.ecdfplot(x='n_days', data=nt.groupby('pid').mean().reset_index(), ax=ax, label='Patient')
    ax.plot([day_cutoff, day_cutoff], [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--')
    ax.legend(bbox_to_anchor=(1.01, 1))

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')

    return nt

def ppdata_fromdict2dict(data, out_file=None, truncate_at_day=7, leading_pad=True, drop_first_day=True, verbose=True):
    '''Create a new datapkl that has measurements for cohort.

    Arguments:
      truncate_at_day (int): (optional, Default=None) in units of d, truncate after x days. None keeps all data
      leading_pad (bool): (optional, Default=True) add pads to align ts so first time is 12 AM
      drop_first_day (bool): (optional, Default=True) drop the first day since, worst case is measurement
        starting at 12 AM
      data_out: pkl file to store measurements for cohort
      pp_md_out: where to store information on cohort selection
    '''

    ppdata = {}

    # criterion
    keys2keep = []
    k_multi_tdeltas = []
    k_ltcutoff = []
    k_lt1d = []
    timer = wearutils.timer()

    if verbose:
        timer.start()
    for i, (k, v) in enumerate(data.items()):
        pass_filters = [True]
        ts, x = v[0], v[1]
        if leading_pad:
            x = np.concatenate((np.zeros((ts[0].hour*60 + ts[0].minute + 1)), [float(i) for i in x]))
            ts = pd.to_datetime(np.linspace(ts[0].replace(hour=0, minute=0).value, ts[0].value, ts[0].hour*60 + ts[0].minute + 1)).to_list() + ts

        # filter: drop first day
        if drop_first_day:
            # drop 1d* 24h/d * 60min/h timepts
            idx = 1*24*60+1
            if not len(ts) <= idx:
                x = x[idx:]
                ts = ts[idx:]
            else:
                pass_filters.append(False)
                k_lt1d.append(k)

        # filter: truncate after X days (post-drop first day, e.g., first week after dropping first day)
        if truncate_at_day is not None:
            idx_end = truncate_at_day*24*60+1
            if not len(ts) < idx_end:
                x = x[:idx_end]
                ts = ts[:idx_end]
            else:
                pass_filters.append(False)
                k_ltcutoff.append(k)

        # filter: if there are measurements that aren't measured every minute, drop
        unique_tdeltas = np.unique(np.diff(pd.Series(ts)))
        if np.timedelta64(1, 'm') not in unique_tdeltas and unique_tdeltas.shape[0] != 1:
            k_multi_tdeltas.append(k)
            pass_filters.append(False)

        # retention
        if all(pass_filters):
            keys2keep.append(k)
            ppdata[k] = (ts, x)

        if verbose:
            if i % 500 == 0 and i != 0 :
                print('  through m={}\t{:.0f}-s elapsed'.format(i+1, timer.stop()))

    if verbose:
        print('\n... finished filtering.')
        print('\nDataset numbers:')
        print('----------------')
        npt, nobs = keys2_npt_nobs(keys2keep)
        print('retained : {}-pts across {}-measurements'.format(npt, nobs))
        npt, nobs = keys2_npt_nobs(k_lt1d)
        print('<1d data : {}-pts across {}-measurements'.format(npt, nobs))
        npt, nobs = keys2_npt_nobs(k_ltcutoff)
        print('<{}d data : {}-pts across {}-measurements'.format(truncate_at_day, npt, nobs))
        npt, nobs = keys2_npt_nobs(k_multi_tdeltas)
        print('multi_dt : {}-pts across {}-measurements'.format(npt, nobs))

    # save
    if out_file is not None:
        with open(out_file, 'wb') as f:
            pickle.dump(ppdata, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        if verbose:
            print('\n wrote pre-processed datapkl to: {}'.format(out_file))

    return ppdata

def chk_datashape(ppdata):
    ts_mismatch = []
    datalen = []
    for k, v in ppdata.items():
        if len(v[0]) != len(v[1]):
            ts_mismatch.append(k)
        datalen.append(len(v[0]))
    if len(np.unique(datalen)) != 1:
        print('Test failed. There are data of different shapes')
    return datalen


def load_pp_rawdata(pp_pkl_out='/home/ngr/gdrive/wearables/data/processed/pp_GAactigraphy.pkl', load_datadict=True):
    '''v0.2 pre-processing of the data from raw download.

    TODO:
      - [x] add arguments for fx calls.
      - [ ] clean run on lab server
    '''
    if load_datadict:
        data = load_datadict()
    else:
        data = pkldata(pkl_out='/home/ngr/gdrive/wearables/data/processed/GAactigraphy_datadict.pkl') # iterative saving

    # quality control
    results_table = chk_tdeltas_multiday(data, out_file='/home/ngr/gdrive/wearables/results/npts_nmeas_after_truncating_longdata.csv')
    nt = get_ntimepoints(data)
    viz_prop_n_days(data, out_file='/home/ngr/gdrive/wearables/results/n_days_after_adding_leadingpad.pdf')

    ## check how many 0s there may be

    ## get data
    ppdata = ppdata_fromdict2dict(data, out_file='/home/ngr/gdrive/wearables/data/processed/ppdata_1wk.pkl')
    datalens = weardata.chk_datashape(ppdata)

    return ppdata

# v0.2
def load_ppdata(filepath='/home/ngr/gdrive/wearables/data/processed/ppdata_1wk.pkl'):
    with open(filepath, 'rb') as f:
        ppdata = pickle.load(f)
        f.close()
    return ppdata

def load_rawmd(filepath='/home/ngr/gdrive/wearables/data/raw/MOD_Data_2021.csv'):
    return pd.read_csv(filepath, low_memory=False)

# v0.1
def load_pp_actigraphy(fname='/home/ngr/gdrive/wearables/data/processed/MOD_1000_Woman_Activity_Data.pkl'):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data

def split_pp_actigraphy(data):
    pids = np.unique([i.split('-')[0] for i in data.keys()])
    train_pids = np.random.choice(pids, int(len(pids)*0.8), replace=False)
    test_pids = [i for i in pids if i not in train_pids]

    train_keys, test_keys = [], []
    for k in data.keys():
        if k.split('-')[0] in train_pids:
            train_keys.append(k)
        elif k.split('-')[0] in test_pids:
            test_keys.append(k)
        else:
            print('{} dict key not found or sorted')

    return {k:data[k] for k in train_keys}, {k:data[k] for k in test_keys}

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

# preoprocess metadata
def nan2n(x_i, n=7):
    x_i = x_i.replace([np.nan], int(n))
    return x_i

def mean_impute(x_i):
    x_i = x_i.fillna(x_i.mean())
    return x_i

def n2nan(x_i, mean_impute_after=True, n=-99):
    x_i = x_i.replace([int(n)], np.nan)
    if mean_impute_after:
        x_i = mean_impute(x_i)
    return x_i

def md_filters(x_i, f):
    if 'nan2' in f:
        x_i = nan2n(x_i, n=f.split('nan2')[1])
    elif '2nan' in f:
        x_i = n2nan(x_i, n=f.split('2nan')[0])
    elif f == 'mean_impute': # technically erroneous because need train/test split info
        x_i = mean_impute(x_i)
    else:
        warnings.warn('Warning. Transform not recognized')
        print('  \nTransformation for {} variable skipped.\n'.format(x_i.name))
    return x_i

def md_data_keymatch(data, md, fix_key=True, del_unmatched_data=True):
    if fix_key:
        # fix a particular key:
        try:
            data['1276-20'] = data['p1276_2014-20']
            del data['p1276_2014-20']
        except KeyError:
            pass
    pid = [k.split('-')[0] for k in data.keys()]
    if del_unmatched_data:
        for k in [i for i in data.keys() if i.split('-')[0] not in md['record_id'].astype(str).to_list()]:
            warnings.warn('Warning. Data-metadata mismatch on key')
            print('  removed {} since it is not in metadata'.format(k))
            del data[k]
    return data, md.loc[md['record_id'].astype(str).isin(pid), :]

def pp_metadata(md, voi, pids2keep=None, out_file=None):
    '''Pre-process metadata and store in pkl with dataframe and variable info.

    TODO:
      - (enhancement): add more lists of transforms, e.g., ['nan2-99', 'lt182dob', 'logpseudocount']
        applied in series

    Arguments:
      md (pd.DataFrame): metadata read in with record_id as pid from csv file
      voi (dict): keys are variable name in metadata and values are tuples, specifying
        (transform, dtype) where transform is a string for a function and dtype is categorical
        or continuous. All continuous var will be stored as np.float32, and categorical is a flag
        to later one-hot-encode (non-ordinal numbers can still be stored). Transform can have
    '''
    if pids2keep is not None:
        # filter out erroneous data by pid (alternatively, metadata may already be filtered)
        md = md.loc[md['record_id'].isin(pids2keep), :]

    ppmd = pd.DataFrame()
    ppmd['record_id'] = md['record_id']

    for i, (k, v) in enumerate(voi.items()):
        x_i = md[k]
        if v[0] is not None:
            if isinstance(v[0], list):
                for f in v[0]:
                    x_i = md_filters(x_i, f)
            else:
                x_i = md_filters(x_i, v[0])
        # type to save
        if v[1] == 'categorical':
            x_i = x_i.astype(str)
        elif v[1] == 'continuous':
            x_i = x_i.astype(np.float32) # single precision
        else:
            warnings.warn('Warning. Date type not recognized')
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

def load_data_md(datapkl_file='/home/ngr/gdrive/wearables/data/processed/ppdata_1wk.pkl',
                 md_file='/home/ngr/gdrive/wearables/data/raw/MOD_Data_2021.csv'):
    '''Load filtered cohort and metadata columns by PID for modeling.

    NOTE:
      - updated 21.08.01
      - the two files indicated are the minimal package for data transfer
    '''
    data = load_ppdata(filepath=datapkl_file)
    # load metadata
    md = load_rawmd(filepath=md_file)
    ppdata, md = md_data_keymatch(data, md)
    voi = {# demographics
            'age_enroll': (['22nan', 'mean_impute'], 'continuous'),
            'marital': ('nan27', 'categorical'),
            'gestage_by': ('nan2-99', 'categorical'),
            'insur': ('nan2-99', 'categorical'),
            'ethnicity': ('nan23', 'categorical'),
            'race': ('nan27', 'categorical'),
            'bmi_1vis': ('mean_impute', 'continuous'),
            'prior_ptb_all': ('nan25', 'categorical'),
            'fullterm_births': ('nan25', 'categorical'),
            'surghx_none': ('nan20', 'categorical'),
            'alcohol': ('nan22', 'categorical'),
            'smoke': ('nan22', 'categorical'),
            'drugs': ('nan22', 'categorical'),
            'hypertension': ('nan22', 'categorical'),
            'pregestational_diabetes': ('nan22', 'categorical'),

            # chronic conditions (?)
            'asthma_yes___1': (None, 'categorical'), # asthma
            'asthma_yes___2': (None, 'categorical'), # diabetes
            'asthma_yes___3': (None, 'categorical'), # gestational hypertension
            'asthma_yes___4': (None, 'categorical'), # CHTN
            'asthma_yes___5': (None, 'categorical'), # anomaly
            'asthma_yes___6': (None, 'categorical'), # lupus
            'asthma_yes___7': (None, 'categorical'), # throid disease
            'asthma_yes___8': (None, 'categorical'), # heart disease
            'asthma_yes___9': (None, 'categorical'), # liver disease
            'asthma_yes___10': (None, 'categorical'), # renal disease
            'asthma_yes___13': (None, 'categorical'), # IUGR
            'asthma_yes___14': (None, 'categorical'), # polyhraminios
            'asthma_yes___15': (None, 'categorical'), # oligohydraminos
            'asthma_yes___18': (None, 'categorical'), # anxiety
            'asthma_yes___19': (None, 'categorical'), # depression
            'asthma_yes___20': (None, 'categorical'), # anemia
            'other_disease': ('nan22', 'categorical'),
            'gestational_diabetes': ('nan22', 'categorical'),
            'ghtn': ('nan22', 'categorical'),
            'preeclampsia': ('nan22', 'categorical'),
            'rh': ('nan22', 'categorical'),
            'corticosteroids': ('nan22', 'categorical'),
            'abuse': ('nan23', 'categorical'),
            'assist_repro': ('nan23', 'categorical'),
            'gyn_infection': ('nan22', 'categorical'),
            'maternal_del_weight': ('-992nan', 'continuous'),
            'ptb_37wks': ('nan22', 'categorical'),

            # vitals and labs @admission
            'cbc_hct': ('-992nan', 'continuous'), # NOTE: some of these shouldn't be negative, need some filtering
            'cbc_wbc': ('-992nan', 'continuous'),
            'cbc_plts': ('-992nan', 'continuous'),
            'cbc_mcv': ('-992nan', 'continuous'),
            'art_ph': ('-992nan', 'continuous'),
            'art_pco2': ('-992nan', 'continuous'),
            'art_po2': ('-992nan', 'continuous'),
            'art_excess': ('-992nan', 'continuous'),
            'art_lactate': ('-992nan', 'continuous'),
            'ven_ph': ('-992nan', 'continuous'),
            'ven_pco2': ('-992nan', 'continuous'),
            'ven_po2': ('-992nan', 'continuous'),
            'ven_excess': ('-992nan', 'continuous'),
            'ven_lactate': ('-992nan', 'continuous'),
            'anes_type': ('-992nan', 'continuous'),
            'epidural': ('nan20', 'categorical'),
            'deliv_mode': ('nan24', 'categorical'),

            # infant things
            'infant_wt': ('-992nan', 'continuous'), # kg
            'infant_length': ('-992nan', 'continuous'),
            'head_circ': ('-992nan', 'continuous'),
            'death_baby': ('nan20', 'categorical'),
            'neonatal_complication': (['22nan', 'nan20'], 'categorical'),

            # postpartum
            'ervisit': ('nan20', 'categorical'),
            'ppvisit_dx': ('nan26', 'categorical'),

            # surveys
            'education1': ('nan2-99', 'categorical'),
            'paidjob1': ('nan20', 'categorical'),
            'work_hrs1': ('nan2-99', 'categorical'),
            'income_annual1': ('nan2-99', 'categorical'),
            'income_support1': ('nan2-99', 'categorical'),
            'regular_period1': ('nan2-88', 'categorical'),
            'period_window1': ('nan2-88', 'categorical'),
            'menstrual_days1': ('nan2-88', 'categorical'),
            'bc_past1': ('nan20', 'categorical'),
            'bc_years1': (['882nan', 'nan2-88'], 'categorical'),
            'months_noprego1': ('nan24', 'categorical'),
            'premature_birth1': ('nan2-88', 'categorical'),
            'stress3_1': ('nan2-99', 'categorical'),
            'workreg_1trim': ('nan20', 'categorical'),

            'choosesleep_1trim': ('nan2-99', 'categorical'),
            'slpwake_1trim': ('nan2-99', 'categorical'),
            'slp30_1trim': ('nan2-99', 'categorical'),
            'sleep_qual1': ('nan2-99', 'categorical'),
            'slpenergy1': ('nan2-99', 'categorical'),
            ## epworth (sum), for interpretation: https://epworthsleepinessscale.com/about-the-ess/ (NOTE: convert 4 to np.nan for sum)
            'sitting1': ('nan20', 'categorical'), ### TODO: add fx to sum this from metadata, then convert to continuous label for regression
            'tv1': ('nan20', 'categorical'),
            'inactive1': ('nan20', 'categorical'),
            'passenger1': ('nan20', 'categorical'),
            'reset1': ('nan20', 'categorical'),
            'talking1': ('nan20', 'categorical'),
            'afterlunch1': ('nan20', 'categorical'),
            'cartraffic1': ('nan20', 'categorical'),
            ## edinburgh depression scale
            'edinb1_1trim': ('nan2-99', 'categorical'),
            'edinb2_1trim': ('nan2-99', 'categorical'),
            'edinb3_1trim': ('nan2-99', 'categorical'),
            'edinb4_1trim': ('nan2-99', 'categorical'),
            'edinb5_1trim': ('nan2-99', 'categorical'),
            'edinb6_1trim': ('nan2-99', 'categorical'),
            'edinb7_1trim': ('nan2-99', 'categorical'),
            'edinb8_1trim': ('nan2-99', 'categorical'),
            'edinb9_1trim': ('nan2-99', 'categorical'),
            'edinb10_1trim': ('nan2-99', 'categorical'),
            ## difficult life circumstances
            ## sleep diary
            }
    ppmd = pp_metadata(md, voi)
    return ppdata, ppmd

# train/test data with user specified label
def split_by_pid(X, df, val=False, label='GA', prop_train=0.8):
    '''Split data by patient ID.
    '''
    pids = np.unique(df['pid'].to_list())
    Y = df[label]
    train_pids = np.random.choice(pids, int(len(pids)*prop_train), replace=False)
    test_pids = [i for i in pids if i not in train_pids]
    train_idx = np.where(df['pid'].isin(train_pids))[0] #df.loc[df['pid'].isin(train_pids), :].index.to_list()
    X_train, y_train = X[train_idx, :], Y.iloc[train_idx]
    if val:
        val_pids = np.random.choice(test_pids, int(len(test_pids)*0.5), replace=False)
        val_idx = np.where(df['pid'].isin(val_pids))[0] #df.loc[df['pid'].isin(val_pids), :].index.to_list()
        X_val, y_val = X[val_idx, :], Y.iloc[val_idx]
        test_pids = [i for i in test_pids if i not in val_pids]
    test_idx = np.where(df['pid'].isin(test_pids))[0] #df.loc[df['pid'].isin(test_pids), :].index.to_list()
    X_test, y_test = X[test_idx, :], Y.iloc[test_idx]
    if val:
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        return X_train, y_train, X_test, y_test

# data transforms
def logpseudocount(X):
    return np.log(X + 1.)

def standardize(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def get_Xy(data, md, val=True, prop_train=0.8, label='GA', transform=True, single_precision=True, verbose=False):
    '''Pick label, and figure out what it is (regression or classification) unless GA.

    TODO:
      - [ ] (enhancement): label encoding before returning data
    '''
    X = np.zeros((len(data.keys()), data[list(data.keys())[0]][1].shape[0]))
    label_df = pd.DataFrame(index=np.arange(len(data.keys())))
    for i, (k, v) in enumerate(data.items()):
        X[i, :] = v[1]
        pid = k.split('-')[0]
        label_df.loc[i, 'pid'] = pid
        label_df.loc[i, 'GA'] = float(k.split('-')[1])
        if label != 'GA':
            # need to add from metadata
            label_df.loc[i, label] = md.loc[md['record_id'].astype(str)==pid, label].values[0]
    if val:
        X_train, y_train, X_val, y_val, X_test, y_test = split_by_pid(X, label_df, val=val, prop_train=prop_train, label=label)
        if transform:
            X_val = standardize(logpseudocount(X_val))
    else:
        X_train, y_train, X_test, y_test = split_by_pid(X, label_df, val=val, prop_train=prop_train, label=label)
    if transform:
        X_train = standardize(logpseudocount(X_train))
        X_test = standardize(logpseudocount(X_test))
    if val:
        if verbose:
            # print shapes
            print('X_train: ({}, {})'.format(X_train.shape[0], X_train.shape[1]))
            print('y_train: ({},)'.format(y_train.shape[0]))
            print('X_val: ({}, {})'.format(X_val.shape[0], X_val.shape[1]))
            print('y_val: ({},)'.format(y_val.shape[0]))
            print('X_test: ({}, {})'.format(X_test.shape[0], X_test.shape[1]))
            print('y_test: ({},)'.format(y_test.shape[0]))
        if single_precision:
            X_train = X_train.astype('float32')
            X_val = X_val.astype('float32')
            X_test = X_test.astype('float32')
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        if verbose:
            # print shapes
            print('X_train: ({}, {})'.format(X_train.shape[0], X_train.shape[1]))
            print('y_train: ({},)'.format(y_train.shape[0]))
            print('X_test: ({}, {})'.format(X_test.shape[0], X_test.shape[1]))
            print('y_test: ({},)'.format(y_test.shape[0]))
        if single_precision:
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
        return X_train, y_train, X_test, y_test

def series2modeltable(y_dict, wide=False, verbose=False):
    '''Use the pd.Series data type to one hot encode or convert to continuous

    Arguments:
      y_dict (dict): Aggregate all y's (y_train, y_test, y_val) as a dict,
        allowing for flexible size.
      wide (bool): (optional, Default=False) whether to leave the y array in
        long or wide form. Ignored if input ys are continuous.
    '''
    # make sure get all for label encoder
    for i, k in enumerate(y_dict.keys()):
        if i==0:
            Y = y_dict[k]
        else:
            Y = Y.append(y_dict[k])
    if Y.dtype == 'float':
        for k in y_dict.keys():
            y_dict[k] = y_dict[k].astype('float32').to_numpy()
        return y_dict, 'regression'
    else:
        # one hot encode
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(Y)
        if verbose:
            print('Categorical label encoding:')
            for i, c in enumerate(le.classes_):
                print('{}: {}'.format(i, c))
        for k in y_dict.keys():
            y_dict[k] = le.transform(y_dict[k]) # converts to int64
            if wide:
                a = np.zeros((y_dict[k].shape[0], Y.max()+1), dtype=int)
                y_dict[k] = a[np.arange(y_dict[k].shape[0]), y_dict[k]] = 1
        return y_dict, le.classes_


if __name__ == '__main__':
    data, md = load_data_md()
    X_train, y_train, X_val, y_val, X_test, y_test = get_Xy(data, md, label='GA', prop_train=0.8, verbose=True)

    # use the data type to convert y to model-ready, then int v. float can determine classification v. regression
    y_dict, target_id = series2modeltable({'y_train':y_train, 'y_val':y_val, 'y_test':y_test})
