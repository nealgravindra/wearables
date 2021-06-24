import pickle
import numpy as np
import pandas as pd

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

def load_actigraphy_metadata(fname='/home/ngr/gdrive/wearables/data/MOD_Data_2021.csv'):
    return pd.read_csv(fname, index_col=0)
