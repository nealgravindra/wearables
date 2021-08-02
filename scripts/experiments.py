import pandas as pd
import pickle
import time
import importlib
import numpy as np
import importlib
import datetime


import sys
sys.path.append('/home/ngr/gdrive/wearables/scripts')
import data as weardata
import utils as wearutils

import matplotlib.pyplot as plt
import seaborn as sns

def pred_all(n_trials=1, model='RF', out_file=None):
    '''Predict GA and other vars in metadata with non-DL models.

    TODO:
      - [ ] (invalid) when grabbing different labels, don't change up indices (currently patients are swapped around each trial)
      - [ ] (invalid) add KFold CV or a way to use val set to actually select  model, as its currently useless
      - [ ] (enhancement) add other models, not just RF

    Arguments:
      model (str): implemented models are RFts (short for RF Time Series).
    '''
    # load data
    timer = wearutils.timer()
    timer.start()
    data, md = weardata.load_data_md()
    print('Data and md loaded in {:.0f}-s'.format(timer.stop()))

    results = pd.DataFrame()
    label_list = ['GA'] + [i for i in md.columns if i!='record_id']
    timer = wearutils.timer()
    for n in range(n_trials):
        for i, target in enumerate(label_list):
            timer.start()
            X_train, y_train, X_val, y_val, X_test, y_test = weardata.get_Xy(data, md, label=target, prop_train=0.8, verbose=False)

            # use the data type to convert y to model-ready, then int v. float can determine classification v. regression
            y_dict, target_id = weardata.series2modeltable({'y_train':y_train, 'y_val':y_val, 'y_test':y_test}, wide=True)
            y_train, y_val, y_test = y_dict['y_train'], y_dict['y_val'], y_dict['y_test']

            if target_id == 'regression':
                from sklearn.metrics import mean_absolute_error
                if model == 'RF':
                    from sktime.regression.interval_based import TimeSeriesForestRegressor
                    from sktime.utils.data_processing import from_2d_array_to_nested
                    X_train_df = from_2d_array_to_nested(X_train)
                    X_val_df = from_2d_array_to_nested(X_val)
                    X_test_df = from_2d_array_to_nested(X_test)

                    model = TimeSeriesForestRegressor()
                    model.fit(X_train_df, y_train)
                    eval_train = mean_absolute_error(y_train, model.predict(X_train_df))
                    eval_val = mean_absolute_error(y_val, model.predict(X_val_df))
                    eval_test = mean_absolute_error(y_test, model.predict(X_test_df))

            else:
                from sklearn.metrics import accuracy_score
                if model == 'RF':
                    from sktime.classification.compose import ComposableTimeSeriesForestClassifier
                    from sktime.utils.data_processing import from_2d_array_to_nested
                    X_train_df = from_2d_array_to_nested(X_train)
                    X_val_df = from_2d_array_to_nested(X_val)
                    X_test_df = from_2d_array_to_nested(X_test)

                    model = ComposableTimeSeriesForestClassifier()
                    model.fit(X_train_df, y_train)
                    eval_train = mean_absolute_error(y_train, model.predict(X_train_df))
                    eval_val = mean_absolute_error(y_val, model.predict(X_val_df))
                    eval_test = mean_absolute_error(y_test, model.predict(X_test_df))

            # store results
            dt = pd.DataFrame({'exp':'pred_all',
                               'model':'TimeSeriesForest',
                               'task':target,
                               'target_id':None,
                               'n_trial':n+1,
                               'eval_train':eval_train,
                               'eval_val':eval_val,
                               'eval_test':eval_test,}, index=[0])
            dt.at[0, 'target_id'] = target_id
            results = results.append(dt, ignore_index=True)
            print('label #{}: {}\tn_trial: {}train: {}\tval: {}\ttest: {}\t{:.0f}-s'.format(i+1, target, n+1, eval_train, eval_val, eval_test, timer.stop()))
        print('\nn_trial={} of {} completed. Time elapsed: {:.1f}-min'.format(j+1, n_trials, timer.sum()/60))

    if out_file is not None:
        results.to_csv(out_file)
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Experiment to initiate')

    args = parser.parse_args()
    exp = args.exp

    if exp == 'all_RF':
        results = pred_all(out_file='/home/ngr/gdrive/wearables/results/all_RF_{}.csv'.format(datetime.datetime.now().strftime('%y%m%d')))
    else:
        print('Program to run experiment does not exist. Not implemented.')
