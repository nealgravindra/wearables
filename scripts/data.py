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

def getdata_fromdict(data, data_out=None, pp_md_out=None, truncate_at_day=7, leading_pad=True, drop_first_day=True, verbose=True):
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

def load_pp_rawdata(pp_pkl_out='/home/ngr/gdrive/wearables/data/processed/pp_GAactigraphy.pkl', load_datadict=True):
    if load_datadict:
        data = load_datadict()
    else:
        data = pkldata(pkl_out='/home/ngr/gdrive/wearables/data/processed/GAactigraphy_datadict.pkl') # iterative saving

    # quality control
    chk_tdeltas_multiday()
    get_ntimepoints()
    viz_prop_n_days()

    ## check how many 0s there may be

    ## get data
    getdata_fromdict()
    return data

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
