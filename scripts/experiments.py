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

def InceptionTime_allmd(exp, trial, colnum):
    # 104 targets
    md_colnames = ['GA', 'age_enroll', 'marital', 'gestage_by', 'insur', 'ethnicity', 
                   'race', 'bmi_1vis', 'prior_ptb_all','fullterm_births', 'surghx_none', 
                   'alcohol', 'smoke', 'drugs', 'hypertension', 'pregestational_diabetes',
                   'asthma_yes___1','asthma_yes___2', 'asthma_yes___3', 'asthma_yes___4', 
                   'asthma_yes___5', 'asthma_yes___6', 'asthma_yes___7', 'asthma_yes___8', 
                   'asthma_yes___9', 'asthma_yes___10','asthma_yes___13', 'asthma_yes___14', 
                   'asthma_yes___15', 'asthma_yes___18', 'asthma_yes___19', 'asthma_yes___20', 
                   'other_disease', 'gestational_diabetes', 'ghtn', 'preeclampsia', 'rh', 
                   'corticosteroids', 'abuse', 'assist_repro', 'gyn_infection', 
                   'maternal_del_weight', 'ptb_37wks', 'cbc_hct', 'cbc_wbc', 'cbc_plts',
                   'cbc_mcv', 'art_ph', 'art_pco2', 'art_po2', 'art_excess', 'art_lactate', 
                   'ven_ph', 'ven_pco2', 'ven_po2', 'ven_excess', 'ven_lactate', 'anes_type', 
                   'epidural', 'deliv_mode', 'infant_wt', 'infant_length', 'head_circ', 
                   'death_baby', 'neonatal_complication', 'ervisit', 'ppvisit_dx', 'education1', 
                   'paidjob1', 'work_hrs1', 'income_annual1', 'income_support1', 'regular_period1', 
                   'period_window1', 'menstrual_days1', 'bc_past1', 'bc_years1', 'months_noprego1', 
                   'premature_birth1', 'stress3_1', 'workreg_1trim', 'choosesleep_1trim', 
                   'slpwake_1trim', 'slp30_1trim', 'sleep_qual1', 'slpenergy1', 'sitting1', 
                   'tv1', 'inactive1', 'passenger1', 'reset1', 'talking1', 'afterlunch1', 
                   'cartraffic1','edinb1_1trim', 'edinb2_1trim', 'edinb3_1trim', 'edinb4_1trim', 
                   'edinb5_1trim', 'edinb6_1trim', 'edinb7_1trim','edinb8_1trim', 'edinb9_1trim', 
                   'edinb10_1trim']
    trainer = weartrain.InceptionTime_trainer(exp=exp, trial=trial,
                                              model_path='/home/ngr4/scratch60/wearables/model_zoo/', 
                                              out_file='/home/ngr4/project/wearables/results/InceptionTimev0.2_allmd.csv',
                                              target=md_colnames[colnum])
    trainer.fit()
    return trainer.eval_test()
    

def pred_death(trial):
    target = 'death_baby'
    trainer = weartrain.InceptionTime_trainer(exp='IT_death', trial=trial,
                                              model_path='/home/ngr4/project/wearables/model_zoo',
                                              out_file='/home/ngr4/project/wearables/results/IT_death.csv', target=target)
    trainer.fit()
    return trainer.eval_test()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Experiment to initiate')
    parser.add_argument('--trial', type=str, help='Relicate number')
    parser.add_argument('--colnum', type=int, help='Column number to specify target from md')

    args = parser.parse_args()
    exp = args.exp
    if args.trial is None:
        trial = 0 # assume dev
    else:
        trial = args.trial
    if args.colnum is not None:
        colnum = args.colnum

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
    elif exp == 'InceptionTimev0.2_allmd':
        res = InceptionTime_allmd(exp, trial, colnum)
        print('Successfully finished exp {}, trial {} for target colnum {}'.format(exp, trial, colnum))
    elif exp == 'IT_death':
        res = pred_death(trial)
    else:
        print('Program to run experiment does not exist. Not implemented.')
