import pandas as pd
import pickle
import time
import importlib
import numpy as np
import importlib
import datetime


import sys
sfp = '/home/ngr4/project/wearables/scripts/'
sys.path.append(sfp)
import data as weardata
import utils as wearutils
import train as weartrain


def pred_all(n_trials=1, model_class='RF', out_file=None, save_train_data=None):
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
    X_train, Y_train, X_val, Y_val, X_test, Y_test = weardata.get_Xy(data, md, prop_train=0.8, verbose=False)
    print('Data and md loaded in {:.0f}-s'.format(timer.stop()))
    if save_train_data is not None:
        with open(save_train_data, 'wb') as f:
            pickle.dump({'X_train':X_train, 'Y_train':Y_train,
                         'X_val':X_val, 'Y_val':Y_val,
                         'X_test':X_test, 'Y_test':Y_test},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

    results = pd.DataFrame()
    label_list = ['GA'] + [i for i in md.columns if i!='record_id']
    timer = wearutils.timer()
    for n in range(n_trials):
        for i, target in enumerate(label_list):
            timer.start()

            # use the data type to convert y to model-ready, then int v. float can determine classification v. regression
            y_dict, target_id = weardata.md2y({'y_train':Y_train, 'y_val':Y_val, 'y_test':Y_test}, label=target, wide=True)
            y_train, y_val, y_test = y_dict['y_train'], y_dict['y_val'], y_dict['y_test']

            if not isinstance(target_id, np.ndarray): # assume it equals regression
                from sklearn.metrics import mean_absolute_error
                if model_class == 'RF':
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
                if model_class == 'knn':
                    from sklearn.neighbors import KNeighborsRegressor
                    model = KNeighborsRegressor(n_neighbors=6)
                    model.fit(X_train, y_train)
                    eval_train = mean_absolute_error(y_train, model.predict(X_train))
                    eval_val = mean_absolute_error(y_val, model.predict(X_val))
                    eval_test = mean_absolute_error(y_test, model.predict(X_test))
            else:
                from sklearn.metrics import accuracy_score
                if model_class == 'RF':
                    from sktime.classification.compose import ComposableTimeSeriesForestClassifier
                    from sktime.utils.data_processing import from_2d_array_to_nested
                    X_train_df = from_2d_array_to_nested(X_train)
                    X_val_df = from_2d_array_to_nested(X_val)
                    X_test_df = from_2d_array_to_nested(X_test)
                    model = ComposableTimeSeriesForestClassifier()
                    model.fit(X_train_df, y_train)
                    eval_train = accuracy_score(y_train, model.predict(X_train_df))
                    eval_val = accuracy_score(y_val, model.predict(X_val_df))
                    eval_test = accuracy_score(y_test, model.predict(X_test_df))
                if model_class == 'knn':
                    from sklearn.neighbors import KNeighborsClassifier
                    model = KNeighborsClassifier(n_neighbors=6)
                    model.fit(X_train, y_train)
                    eval_train = accuracy_score(y_train, model.predict(X_train))
                    eval_val = accuracy_score(y_val, model.predict(X_val))
                    eval_test = accuracy_score(y_test, model.predict(X_test))

            # store results
            dt = pd.DataFrame({'exp':'pred_all',
                               'model':model_class,
                               'task':target,
                               'target_id':None,
                               'n_trial':n+1,
                               'eval_train':eval_train,
                               'eval_val':eval_val,
                               'eval_test':eval_test,}, index=[0])
            dt.at[0, 'target_id'] = target_id
            # store OR del model in dt
            del model
            results = results.append(dt, ignore_index=True)
            print('label #{}: {}\tn_trial: {}\ttrain: {:.2f}\tval: {:.2f}\ttest: {:.2f}\t{:.0f}-s'.format(i+1, target, n+1, eval_train, eval_val, eval_test, timer.stop()))
        print('\nn_trial={} of {} completed. Time elapsed: {:.1f}-min'.format(n+1, n_trials, timer.sum()/60))

    if out_file is not None:
        results.to_csv(out_file)
    return results

def InceptionTimeRegressor(trial, patience=200):
    trainer = weartrain.InceptionTimeRegressor_trainer(model_path='/home/ngr4/project/wearables/model_zoo',
                                                       trial=trial, out_file='/home/ngr4/project/wearables/results/InceptionTimeRegressor_v1_GA.csv',
                                                       patience=patience, n_epochs=2000,
                                                       batch_size=32)
    trainer.fit()
    return trainer.eval_test()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Experiment to initiate')
    parser.add_argument('--trial', type=str, help='Relicate number')

    args = parser.parse_args()
    exp = args.exp
    if args.trial is None:
        trial = 0 # assume dev
    else:
        trial = args.trial

    if exp == 'all_RF':
        results = pred_all(model_class='RF', out_file='/home/ngr/gdrive/wearables/results/all_RF_{}.csv'.format(datetime.datetime.now().strftime('%y%m%d')))
    elif exp == 'all_knn':
        results = pred_all(n_trials=10, model_class='knn',
                           out_file='/home/ngr/gdrive/wearables/results/all_knn_{}.csv'.format(datetime.datetime.now().strftime('%y%m%d')),
                           save_train_data='/home/ngr/gdrive/wearables/data/processed/datapkl_Xactigraphy_Ymd_trainvaltest{}.pkl'.format(datetime.datetime.now().strftime('%y%m%d')))
    elif exp == 'InceptionTime_dev':
        res = InceptionTimeRegressor(trial=trial)
        print('Finished exp {}, trial {}'.format(exp, trial))
    elif exp == 'InceptionTime_nopatience':
        res = InceptionTimeRegressor(trial=trial, patience=None)
    else:
        print('Program to run experiment does not exist. Not implemented.')
