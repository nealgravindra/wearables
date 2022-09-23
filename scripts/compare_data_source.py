import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/ngrav/project')

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics as sklmetrics
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from wearables.scripts import eval_ as weareval
from imblearn.over_sampling import SMOTE
import pickle
import time

class knncv():
    def __init__(self, metadata, voi, embeds=None, trainer=None):
        '''Association by effectiveness of the classifier or coefficients.

        Arguments:
          voi (dict)
          embeds (pd.DataFrame): it
          trainer (obj): 
          metadata (pd.DataFrame): contains labels. Assumes index is unique_id to match with list of ids from embeddings
        '''
        self.metadata = metadata
        self.voi = voi
        self.embeds = embeds
        self.trainer = trainer
        # self.X = self.get_model_X()
        self.get_model_X()
        self.res = {} # 'class_nb': (model, scores)
        self.summary = {}
            
    def get_model_X(self):
        # load data
        if self.embeds is not None:
            self.data = self.embeds
            # filter for cohort or ensure md matches
            self.data = self.data.loc[self.metadata.index, :]
            self.grps = [s.split('_')[0] for s in self.data.index]
        elif self.trainer is not None:
            # error prone, custom implementation
            self.data = pd.DataFrame({uid: self.trainer.data.data['data'][uid]['activity'][:-1].to_numpy() for uid in self.trainer.data.data['IDs']}).T
            # filter for cohort or ensure md matches
            self.data = self.data.loc[self.metadata.index, :]
            self.grps = [s.split('_')[0] for s in self.data.index]
        self.splitter = GroupShuffleSplit(n_splits=5, train_size=0.8, random_state=42)
        # return self.data.to_numpy(dtype=np.float32)
        
    def get_model_y(self, target_name):
        y = self.metadata.loc[self.data.index.to_list(), target_name]
        if 'cat' in self.voi[target_name]: 
            # convert to int
            if len(y.unique()) == 1:
                print(f"\nonly one val for {target_name}")
                print('... cannot fit one class only. Reconsider its inclusion. Skipping this var.')
                return None

            else:
                try:
                    y = y.to_numpy(dtype=int)
                    if len(np.unique(y)) > 2:
                        y = label_binarize(y, classes=np.sort(np.unique(y)))
                    else:
                        y = y.reshape(-1, 1)
                except ValueError:
                    y = label_binarize(y, classes=np.sort(np.unique(y)))
        else:
            y = y.to_numpy(dtype=np.float32)
        return y
    
    def fit(self, target_name, smote=True, verbose=False):
        X, y = self.data.to_numpy(dtype=np.float32), self.get_model_y(target_name)
        # CV splits
        for i, (train_idx, test_idx) in enumerate(self.splitter.split(X, y, self.grps)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            if 'cat' in self.voi[target_name]: # SMOTE
                fpr, tpr = dict(), dict()
                au_roc, eval_metrics = dict(), dict()
                for jj in range(y_train.shape[1]):
                    if smote:
                        oversample = SMOTE(k_neighbors=3)
                        # print(f'kk: {kk}\tX_train: {X_train.shape}\ty_train: {y_train.shape}')
                        try:
                            X_train_mod, y_train_mod = oversample.fit_resample(X_train, y_train[:, jj])
                        except ValueError:
                            print("\n{}-th class cannot be computed. Too few n_samples. Skipping".format(jj)) 
                            if verbose:
                                print('{} class frequencies:'.format(catvar))
                                for jjj in range(y_train.shape[1]):
                                    print(f"j: {jjj}\t0: {(y_train[:, jjj]==0).sum()}\t1: {(y_train[:, jjj]==1).sum()}")
                            continue
                        del oversample
                    else:
                        X_train_mod, y_train_mod = X_train, y_train

                    # model/eval
                    model = KNeighborsClassifier(n_jobs=12)
                    model.fit(X_train_mod, y_train_mod)
                    (fpr[jj], tpr[jj], thresholds) = sklmetrics.roc_curve(y_test[:, jj], model.predict_proba(X_test)[:, 1])
                    au_roc[jj] = sklmetrics.auc(fpr[jj], tpr[jj])
                    eval_metrics[jj] = weareval.eval_output(model.predict(X_test), y_test[:, jj], 
                                                            tasktype='regression' if 'cont' in self.voi[target_name] else 'classification')
                    # scores = cross_val_score(lr, X_train_mod, y_train_mod, cv=5, scoring='roc_curve')
                self.res[f'{target_name}_fold{i}'] = (fpr, tpr, au_roc, eval_metrics) # (lr, scores)
            else:
                model = KNeighborsRegressor(n_jobs=12) 
                model.fit(X_train, y_train)
                self.res[f'{target_name}_fold{i}'] = weareval.eval_output(model.predict(X_test), y_test, 
                                                                          tasktype='regression' if 'cont' in self.voi[target_name] else 'classification')
    
    def fit_all(self, verbose=True):
        if verbose:
            tic_total = time.time()
        for i, (target_name, task_type) in enumerate(self.voi.items()):
            if verbose:
                tic = time.time()
                print('Starting {} ({} of n={} tasks)...'.format(target_name, i+1, len(self.voi.keys())))
            self.fit(target_name)
            if verbose:
                print('... done in {:.2f}-s\t{:.1f}-min elapsed'.format(time.time() - tic, (time.time() - tic_total)/60))
        # generate summary
        self.summarize()
                
    def summarize(self, reg_metric='rho', clf_metric='auroc'):
        '''
        Arguments:
          reg_metric (str): [optional] one of 'rho', 'mae', 'mape'
          clf_metric (str): [optional] one of 'auroc', 'auprc'
        '''
        for i, k in enumerate(self.voi.keys()):
            all_folds = [ii for ii in self.res.keys() if ii.split('_fold')[0]==k]
            if 'cont' in self.voi[k]:
                metric = []
                for kk in all_folds:
                    metric.append(self.res[kk][reg_metric])
            else:
                if clf_metric == 'auroc':
                # macro-average
                    metric = []
                    for kk in all_folds:
                        within_class = []
                        for k_class in self.res[kk][2].keys():
                            within_class.append(self.res[kk][2][k_class])
                        metric.append(np.nanmean(within_class)) # could max?
                elif clf_metric == 'auprc':
                    for kk in all_folds:
                        within_class = []
                        for k_class in self.res[kk][3].keys():
                            within_class.append(self.res[kk][3]['auprc_model'])
                        metric.append(np.nanmean(within_class)) 
                elif clf_metric == 'prc':
                    counter = 0
                    for kk in all_folds:
                        for j, k_class in self.res[kk][0].keys():
                            if counter==0:
                                all_fpr = self.res[kk][0][k_class]
                                counter += 1
                            else:
                                all_fpr = np.concatenate((all_fpr, self.res[kk][0][k_class]))
                                counter += 1
                    all_fpr = np.unique(all_fpr)
                    metric = []
                    counter = 0
                    for kk in all_folds: 
                        for j, n_class in enumerate(range(len(self.res[kk][0].keys()))):
                            if counter == 0:
                                tpr = np.interp(all_fpr, self.res[kk][0][n_class], self.res[kk][1][n_class])
                            else:
                                tpr = np.vstack((tpr, np.interp(all_fpr, self.res[kk][0][n_class], self.res[kk][1][n_class])))
                    metric.append((all_fpr, tpr))
            self.summary[k] = metric

def load_data():
    # add back GA to pred
    from wearables.scripts.md_specification import mdpred_voi
    import wearables.scripts.eval_ as weareval
    pfp = '/home/ngrav/project/wearables/results/'
    mfp = '/home/ngrav/project/wearables/model_zoo'

    # filepaths to bst or pre-processed md with calculated metrics 
    pp_md_fp = os.path.join(pfp, 'md_v522_220124.csv')
    bst_trainer = os.path.join(mfp, 'trainer_itv52_InceptionTime_GA5.pkl')
    bst_modelpkl = os.path.join(mfp, '213-itv52_InceptionTime_GA5.pkl')
    bst_modelembeds = os.path.join(pfp, 'embeds_v522_220124.csv')

    # load up to date md
    md = pd.read_csv(pp_md_fp, index_col=0)
    def loadpkl(fp):
        with open(fp, 'rb') as f:
            return pickle.load(f)
    trainer = loadpkl(bst_trainer)
    it = pd.read_csv(bst_modelembeds, index_col=0)

    # update mdpred_voi to include GA prediction
    mdpred_voi['GA'] = 'continuous'
    return md, it, trainer, mdpred_voi
    
def exp(out_file=None):
    md, embeds, trainer, voi = load_data()
    
    # compare raw to embeddings, results on CV-folds
    knn_raw = knncv(md, voi, trainer=trainer)
    knn_raw.fit_all()
    
    knn_embeds = knncv(md, voi, embeds=embeds)
    knn_embeds.fit_all()
    
    if out_file is not None:
        with open(out_file, 'wb') as f:
            pickle.dump({'knn_raw': (knn_raw.summary, knn_raw.res),
                         'knn_embeds': (knn_embeds.summary, knn_embeds.res)}, 
                        f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
    
    
if __name__ == '__main__':
    exp(out_file='/home/ngrav/project/wearables/results/comp_modelembedsVrawactigraphy_kNN_220321.pkl')
    os.system("mail -s comps_done ngravindra@gmail.com")
    