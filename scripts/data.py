import pickle
import numpy as np
import pandas as pd
import os
import glob
import warnings
import time

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

def load_pp_rawdata(pp_pkl_out='/home/ngr/gdrive/wearables/data/processed/pp_GAactigraphy.pkl'):
    data = pkldata(pkl_out='/home/ngr/gdrive/wearables/data/processed/GAactigraphy_datadict.pkl') # iterative saving
    # quality control
    ## check how many 0s there may be

    ## transformations
    return data

def load_pp_actigraphy(fname='/home/ngr/gdrive/wearables/data/processed/MOD_1000_Woman_Activity_Data.pkl'):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data

def load_datadict(fname='/home/ngr/gdrive/wearables/data/processed/GAactigraphy_datadict.pkl'):
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
