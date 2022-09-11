'''
'model_comp_nonDLmodels.py'

# models implemented
1. kNN
2. kNN-DTW
3. RandomForest
4. LightGBM
5. RFTimeRegressor


Implementation assumptions:
  (1) a decision had to be made as to how to choose the best model over CV folds.
      This was chosen to be MAE for regression tasks and adjusted balanced acc for 
      classification tasks, as the latter is known as informativeness or Youden's J 
      statistic.
  (2) it is assumed that cross validation is called for and used when loading data into numpy 
      format.
'''

import time
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/ngrav/project/')
from wearables.scripts import data as weardata
from wearables.scripts import eval_ as weareval

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import fastdtw
import lightgbm as lgb
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.regression.compose._ensemble import ComposableTimeSeriesForestRegressor
from sktime.classification.compose import ComposableTimeSeriesForestClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

# try to overcome


class kNN():
    def __init__(self, n_trials=10, target_name='GA'):
        self.n_trials = n_trials
        self.target_name = target_name
    
        
    def load_data(self):
        return weardata.dataloader(target_name=self.target_name)
        
    def fit(self, data):
        if data.kfold > 1:
            cv_eval = {}
            for k, cv_fold in enumerate(data.Xy_train.keys()):
                [(X_train, y_train), (X_val, y_val)] = data.Xy_train[cv_fold]
                kNN = KNeighborsRegressor(n_jobs=12) if data.tasktype=='regression' else KNeighborsClassifier(n_jobs=12)
                kNN.fit(X_train, y_train)
                eval_metrics = weareval.eval_output(kNN.predict(X_val), y_val, tasktype=data.tasktype)
                cv_eval[cv_fold] = {'model': kNN, 
                                    # 'data': [(X_train, y_train), (X_val, y_val)], # store just IDs?
                                    'metric': eval_metrics['mae'] if data.tasktype=='regression' else eval_metrics['balanced_acc_adj'],
                                    'metrics': eval_metrics}
            # retain only best model
            tmp = {cv_fold:cv_eval[cv_fold]['metric'] for cv_fold in cv_eval.keys()}
            bst_fold = min(tmp, key=tmp.get) if data.tasktype=='regression' else max(tmp, key=tmp.get)
            self.kNN = cv_eval[bst_fold]['model']
            return {'model': self.kNN, 'metrics': cv_eval[bst_fold]['metrics']}
        else:
            X_train, y_train = data.Xy_train
            X_val, y_val = data.Xy_val
            self.kNN = KNeighborsRegressor(n_jobs=12) if data.tasktype=='regression' else KNeighborsClassifier(n_jobs=12)
            self.kNN.fit(X_train, y_train)
            eval_metrics = weareval.eval_output(self.kNN.predict(X_val), y_val, tasktype=data.tasktype)
            return {'model': self.kNN, 'metrics': eval_metrics}
    
    def eval_test(self, data):
        X_test, y_test = data.Xy_test
        eval_metrics = weareval.eval_output(self.kNN.predict(X_test), y_test, tasktype=data.tasktype)
        return eval_metrics
    
    def del_model(self):
        del self.kNN
    
    def run_trials(self, verbose=True):
        if verbose:
            t_start = time.time()
            print('Starting kNN trials; predict {}'.format(self.target_name))
        results = {}
        for n in range(self.n_trials):
            if verbose:
                tic = time.time()
            data = self.load_data()
            self.fit(data)
            results[n] = self.eval_test(data)
            self.del_model()
            if verbose:
                print('  finished trial {} in {:.2f}-s\t{:.1f}-min elapsed'.format(n, time.time()-tic, (time.time()-t_start)/60))
        return results

# NOTE: this is very slow; slowest step is the prediction step
#   there may be ways to increase it (by embeddings, then mapping)
class kNNDTW():
    def __init__(self, n_trials=10, target_name='GA', downsample_inference=False):
        self.n_trials = n_trials
        self.target_name = target_name
        self.downsample_inference = downsample_inference
        
    def dtw(self, x, y):
        return fastdtw.dtw(x, y)[0]
    
        
    def load_data(self):
        return weardata.dataloader(target_name=self.target_name)
        
    def fit(self, data):
        if data.kfold > 1:
            cv_eval = {}
            for k, cv_fold in enumerate(data.Xy_train.keys()):
                print('    starting kfold=', cv_fold)
                [(X_train, y_train), (X_val, y_val)] = data.Xy_train[cv_fold]
                kNNDTW = KNeighborsRegressor(
                    n_jobs=-1, 
                    algorithm='ball_tree',
                    weights='distance',
                    metric=self.dtw) if data.tasktype=='regression' else KNeighborsClassifier(
                    n_jobs=-1, 
                    algorithm='ball_tree',
                    weights='distance',
                    metric=self.dtw)
                kNNDTW.fit(X_train, y_train)
                if self.downsample_inference:
                    idx = np.random.choice(np.arange(X_val.shape[0]), 20, replace=False)
                    X_val, y_val = X_val[idx], y_val[idx]
                eval_metrics = weareval.eval_output(kNNDTW.predict(X_val), y_val, tasktype=data.tasktype)
                cv_eval[cv_fold] = {'model': kNNDTW, 
                                    # 'data': [(X_train, y_train), (X_val, y_val)], # store just IDs?
                                    'metric': eval_metrics['mae'] if data.tasktype=='regression' else eval_metrics['balanced_acc_adj'],
                                    'metrics': eval_metrics}
            # retain only best model
            tmp = {cv_fold:cv_eval[cv_fold]['metric'] for cv_fold in cv_eval.keys()}
            bst_fold = min(tmp, key=tmp.get) if data.tasktype=='regression' else max(tmp, key=tmp.get)
            self.kNNDTW = cv_eval[bst_fold]['model']
            return {'model': self.kNNDTW, 'metrics': cv_eval[bst_fold]['metrics']}
        else:
            X_train, y_train = data.Xy_train
            X_val, y_val = data.Xy_val
            self.kNNDTW = kNNDTW = KNeighborsRegressor(
                n_jobs=-1, 
                algorithm='ball_tree',
                weights='distance',
                metric=self.dtw) if data.tasktype=='regression' else KNeighborsClassifier(
                n_jobs=-1, 
                algorithm='ball_tree',
                weights='distance',
                metric=self.dtw)
            self.kNNDTW.fit(X_train, y_train)
            if self.downsample_inference:
                idx = np.random.choice(np.arange(X_val.shape[0]), 20, replace=False)
                X_val, y_val = X_val[idx], y_val[idx]
            eval_metrics = weareval.eval_output(self.kNNDTW.predict(X_val), y_val, tasktype=data.tasktype)
            return {'model': self.kNNDTW, 'metrics': eval_metrics}
    
    def eval_test(self, data):
        X_test, y_test = data.Xy_test
        if self.downsample_inference:
            idx = np.random.choice(np.arange(X_test.shape[0]), 20, replace=False)
            X_test, y_test = X_test[idx], y_test[idx]
        eval_metrics = weareval.eval_output(self.kNNDTW.predict(X_test), y_test, tasktype=data.tasktype)
        return eval_metrics
    
    def del_model(self):
        del self.kNNDTW
    
    def run_trials(self, verbose=True):
        if verbose:
            t_start = time.time()
            print('Starting kNNDTW trials; predict {}'.format(self.target_name))
        results = {}
        for n in range(self.n_trials):
            if verbose:
                tic = time.time()
            data = self.load_data()
            self.fit(data)
            results[n] = self.eval_test(data)
            self.del_model()
            if verbose:
                print('  finished trial {} in {:.2f}-s\t{:.1f}-min elapsed'.format(n, time.time()-tic, (time.time()-t_start)/60))
        return results
        
class RandomForest():
    '''
    Arguments:
      data (weardata class): dataloader class from data.py
    '''
    def __init__(self, n_trials=10, target_name='GA'):
        self.n_trials = n_trials
        self.target_name = target_name
    
        
    def load_data(self):
        return weardata.dataloader(target_name=self.target_name)
        
    def fit(self, data):
        if data.kfold > 1:
            cv_eval = {}
            for k, cv_fold in enumerate(data.Xy_train.keys()):
                [(X_train, y_train), (X_val, y_val)] = data.Xy_train[cv_fold]
                RF = RandomForestRegressor(n_jobs=12) if data.tasktype=='regression' else RandomForestClassifier(n_jobs=12)
                RF.fit(X_train, y_train)
                eval_metrics = weareval.eval_output(RF.predict(X_val), y_val, tasktype=data.tasktype)
                cv_eval[cv_fold] = {'model': RF, 
                                    # 'data': [(X_train, y_train), (X_val, y_val)], # store just IDs?
                                    'metric': eval_metrics['mae'] if data.tasktype=='regression' else eval_metrics['balanced_acc_adj'],
                                    'metrics': eval_metrics}
            # retain only best model
            tmp = {cv_fold:cv_eval[cv_fold]['metric'] for cv_fold in cv_eval.keys()}
            bst_fold = min(tmp, key=tmp.get) if data.tasktype=='regression' else max(tmp, key=tmp.get)
            self.RF = cv_eval[bst_fold]['model']
            return {'model': self.RF, 'metrics': cv_eval[bst_fold]['metrics']}
        else:
            X_train, y_train = data.Xy_train
            X_val, y_val = data.Xy_val
            self.RF = RandomForestRegressor(n_jobs=12) if data.tasktype=='regression' else RandomForestClassifier(n_jobs=12)
            self.RF.fit(X_train, y_train)
            eval_metrics = weareval.eval_output(self.RF.predict(X_val), y_val, tasktype=data.tasktype)
            return {'model': self.RF, 'metrics': eval_metrics}
    
    def eval_test(self, data):
        X_test, y_test = data.Xy_test
        eval_metrics = weareval.eval_output(self.RF.predict(X_test), y_test, tasktype=data.tasktype)
        return eval_metrics
    
    def del_model(self):
        del self.RF
    
    def run_trials(self, verbose=True):
        if verbose:
            t_start = time.time()
            print('Starting RandomForest trials; predict {}'.format(self.target_name))
        results = {}
        for n in range(self.n_trials):
            if verbose:
                tic = time.time()
            data = self.load_data()
            self.fit(data)
            results[n] = self.eval_test(data)
            self.del_model()
            if verbose:
                print('  finished trial {} in {:.2f}-s\t{:.1f}-min elapsed'.format(n, time.time()-tic, (time.time()-t_start)/60))
        return results
    
class LightGBM():
    '''
    Arguments:
      data (weardata class): dataloader class from data.py
    '''
    def __init__(self, n_trials=10, target_name='GA'):
        self.n_trials = n_trials
        self.target_name = target_name
    
        
    def load_data(self):
        return weardata.dataloader(target_name=self.target_name)
        
    def fit(self, data):
        params = {
            'boosting_type': 'gbdt',
            'verbosity': 0} 
        if data.tasktype == 'regression':    
            params['objective'] = 'regression',
        else:
            if len(data.Xy_test[1].shape) > 1:
                params['objective'] = 'multiclass',
            else:
                params['objective'] = 'binary',
        if data.kfold > 1:
            cv_eval = {}
            for k, cv_fold in enumerate(data.Xy_train.keys()):
                [(X_train, y_train), (X_val, y_val)] = data.Xy_train[cv_fold]
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_val, y_val)
                gbm = lgb.train(params,
                                lgb_train, 
                                valid_sets=lgb_eval,
                                callbacks=[lgb.early_stopping(stopping_rounds=5)])
                eval_metrics = weareval.eval_output(gbm.predict(X_val, num_iteration=gbm.best_iteration), y_val, tasktype=data.tasktype)
                cv_eval[cv_fold] = {'model': gbm, 
                                    # 'data': [(X_train, y_train), (X_val, y_val)], # store just IDs?
                                    'metric': eval_metrics['mae'] if data.tasktype=='regression' else eval_metrics['balanced_acc_adj'],
                                    'metrics': eval_metrics}
            # retain only best model
            tmp = {cv_fold:cv_eval[cv_fold]['metric'] for cv_fold in cv_eval.keys()}
            bst_fold = min(tmp, key=tmp.get) if data.tasktype=='regression' else max(tmp, key=tmp.get)
            self.gbm = cv_eval[bst_fold]['model']
            return {'model': self.gbm, 'metrics': cv_eval[bst_fold]['metrics']}
        else:
            X_train, y_train = data.Xy_train
            X_val, y_val = data.Xy_val
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_val, y_val)
            self.gbm = lgb.train(params,
                            lgb_train, 
                            valid_sets=lgb_eval,
                            callbacks=[lgb.early_stopping(stopping_rounds=5)])
            eval_metrics = weareval.eval_output(self.gbm.predict(X_val, num_iteration=gbm.best_iteration), y_val, tasktype=data.tasktype)
            return {'model': self.gbm, 'metrics': eval_metrics}
    
    def eval_test(self, data):
        X_test, y_test = data.Xy_test
        eval_metrics = weareval.eval_output(self.gbm.predict(X_test), y_test, tasktype=data.tasktype)
        return eval_metrics
    
    def del_model(self):
        del self.gbm
    
    def run_trials(self, verbose=True):
        if verbose:
            t_start = time.time()
            print('Starting LightGBM trials; predict {}'.format(self.target_name))
        results = {}
        for n in range(self.n_trials):
            if verbose:
                tic = time.time()
            data = self.load_data()
            self.fit(data)
            results[n] = self.eval_test(data)
            self.del_model()
            if verbose:
                print('  finished trial {} in {:.2f}-s\t{:.1f}-min elapsed'.format(n, time.time()-tic, (time.time()-t_start)/60))
        return results
    
class TimeSeriesForest():
    '''
    Arguments:
      data (weardata class): dataloader class from data.py
    '''
    def __init__(self, n_trials=10, target_name='GA'):
        self.n_trials = n_trials
        self.target_name = target_name
    
        
    def load_data(self):
        return weardata.dataloader(target_name=self.target_name)
        
    def fit(self, data):
        if data.kfold > 1:
            cv_eval = {}
            for k, cv_fold in enumerate(data.Xy_train.keys()):
                [(X_train, y_train), (X_val, y_val)] = data.Xy_train[cv_fold]
                X_train, X_val = from_2d_array_to_nested(X_train), from_2d_array_to_nested(X_val)
                tsf = TimeSeriesRegressor(
                    n_jobs=-1) if data.tasktype=='regression' else ComposableTimeSeriesForestClassifier(
                    n_jobs=-1)
                tsf.fit(X_train, y_train)
                eval_metrics = weareval.eval_output(tsf.predict(X_val), y_val, tasktype=data.tasktype)
                cv_eval[cv_fold] = {'model': tsf, 
                                    # 'data': [(X_train, y_train), (X_val, y_val)], # store just IDs?
                                    'metric': eval_metrics['mae'] if data.tasktype=='regression' else eval_metrics['balanced_acc_adj'],
                                    'metrics': eval_metrics}
            # retain only best model
            tmp = {cv_fold:cv_eval[cv_fold]['metric'] for cv_fold in cv_eval.keys()}
            bst_fold = min(tmp, key=tmp.get) if data.tasktype=='regression' else max(tmp, key=tmp.get)
            self.tsf = cv_eval[bst_fold]['model']
            return {'model': self.tsf, 'metrics': cv_eval[bst_fold]['metrics']}
        else:
            X_train, y_train = data.Xy_train
            X_val, y_val = data.Xy_val
            X_train, X_val = from_2d_array_to_nested(X_train), from_2d_array_to_nested(X_val)
            self.tsf = TimeSeriesRegressor(
                n_jobs=-1) if data.tasktype=='regression' else ComposableTimeSeriesForestClassifier(
                n_jobs=-1)
            self.tsf.fit(X_train, y_train)
            eval_metrics = weareval.eval_output(self.tsf.predict(X_val), y_val, tasktype=data.tasktype)
            return {'model': self.tsf, 'metrics': eval_metrics}
    
    def eval_test(self, data):
        X_test, y_test = data.Xy_test
        X_test = from_2d_array_to_nested(X_test)
        eval_metrics = weareval.eval_output(self.tsf.predict(X_test), y_test, tasktype=data.tasktype)
        return eval_metrics
    
    def del_model(self):
        del self.tsf
    
    def run_trials(self, verbose=True):
        if verbose:
            t_start = time.time()
            print('Starting TimeSeriesForest trials; predict {}'.format(self.target_name))
        results = {}
        for n in range(self.n_trials):
            if verbose:
                tic = time.time()
            data = self.load_data()
            self.fit(data)
            results[n] = self.eval_test(data)
            self.del_model()
            if verbose:
                print('  finished trial {} in {:.2f}-s\t{:.1f}-min elapsed'.format(n, time.time()-tic, (time.time()-t_start)/60))
        return results

# slow, even after sktime update (to v0.9.0 with numba incorporation)
class kNNTS():
    '''
    Arguments:
      data (weardata class): dataloader class from data.py
    '''
    def __init__(self, n_trials=10, target_name='GA'):
        self.n_trials = n_trials
        self.target_name = target_name
    
        
    def load_data(self):
        return weardata.dataloader(target_name=self.target_name)
        
    def fit(self, data):
        if data.kfold > 1:
            cv_eval = {}
            for k, cv_fold in enumerate(data.Xy_train.keys()):
#                 print('    cv_fold: ', cv_fold)
                [(X_train, y_train), (X_val, y_val)] = data.Xy_train[cv_fold]
                X_train, X_val = from_2d_array_to_nested(X_train), from_2d_array_to_nested(X_val)
                knn = KNeighborsTimeSeriesClassifier(n_neighbors=5, distance="dtw", n_jobs=-1)
                knn.fit(X_train, y_train)
                eval_metrics = weareval.eval_output(knn.predict(X_val), y_val, tasktype=data.tasktype)
                cv_eval[cv_fold] = {'model': knn, 
                                    # 'data': [(X_train, y_train), (X_val, y_val)], # store just IDs?
                                    'metric': eval_metrics['mae'] if data.tasktype=='regression' else eval_metrics['balanced_acc_adj'],
                                    'metrics': eval_metrics}
            # retain only best model
            tmp = {cv_fold:cv_eval[cv_fold]['metric'] for cv_fold in cv_eval.keys()}
            bst_fold = min(tmp, key=tmp.get) if data.tasktype=='regression' else max(tmp, key=tmp.get)
            self.knn = cv_eval[bst_fold]['model']
            return {'model': self.knn, 'metrics': cv_eval[bst_fold]['metrics']}
        else:
            X_train, y_train = data.Xy_train
            X_val, y_val = data.Xy_val
            X_train, X_val = from_2d_array_to_nested(X_train), from_2d_array_to_nested(X_val)
            self.knn = knn = KNeighborsTimeSeriesClassifier(n_neighbors=5, distance="dtw", n_jobs=-1)
            self.knn.fit(X_train, y_train)
            eval_metrics = weareval.eval_output(self.knn.predict(X_val), y_val, tasktype=data.tasktype)
            return {'model': self.knn, 'metrics': eval_metrics}
    
    def eval_test(self, data):
        X_test, y_test = data.Xy_test
        X_test = from_2d_array_to_nested(X_test)
        eval_metrics = weareval.eval_output(self.knn.predict(X_test), y_test, tasktype=data.tasktype)
        return eval_metrics
    
    def del_model(self):
        del self.knn
    
    def run_trials(self, verbose=True):
        if verbose:
            t_start = time.time()
            print('Starting kNTimeSeries trials; predict {}'.format(self.target_name))
        results = {}
        for n in range(self.n_trials):
            if verbose:
                tic = time.time()
            data = self.load_data()
            self.fit(data)
            results[n] = self.eval_test(data)
            self.del_model()
            if verbose:
                print('  finished trial {} in {:.2f}-s\t{:.1f}-min elapsed'.format(n, time.time()-tic, (time.time()-t_start)/60))
        return results
    
def exps_GA(n_trials=10, out_file=None):
    overall_results = pd.DataFrame()
    
    # kNN
    results = kNN(n_trials=n_trials).run_trials()
    results = pd.DataFrame(results).T
    results['model'] = 'kNN'
    overall_results = overall_results.append(results)
    
    if out_file is not None:
        overall_results.to_csv(out_file)
    
    # RandomForest
    results = RandomForest(n_trials=n_trials).run_trials()
    results = pd.DataFrame(results).T
    results['model'] = 'RandomForest'
    overall_results = overall_results.append(results)
    
    if out_file is not None:
        overall_results.to_csv(out_file)
    
    # Gradient Boosting
    results = LightGBM(n_trials=n_trials).run_trials()
    results = pd.DataFrame(results).T
    results['model'] = 'Gradient Boosting (LightGBM)'
    overall_results = overall_results.append(results)
    
    if out_file is not None:
        overall_results.to_csv(out_file)
    
#     # TimeSeriesForest
        # NOTE: there is an error in method definition of the sktime class now, preventing the run
#     results = TimeSeriesForest(n_trials=n_trials).run_trials()
#     results = pd.DataFrame(results).T
#     results['model'] = 'TimeSeriesForest'
#     overall_results = overall_results.append(results)
    
#     if out_file is not None:
#         overall_results.to_csv(out_file)
        
    # kNN-DTW, hog it
    results = kNNDTW(n_trials=3).run_trials()
    results = pd.DataFrame(results).T
    results['model'] = 'kNN-DTW'
    overall_results = overall_results.append(results)
    
    if out_file is not None:
        overall_results.to_csv(out_file)
    
    return overall_results

if __name__ == '__main__':
    results = exps_GA(n_trials=10, out_file='/home/ngrav/project/wearables/results/model_cmp_nonDL_v71.csv')