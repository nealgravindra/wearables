import os
import sys
sfp = '/home/ngr4/project/'
sys.path.append(sfp)
from wearables.scripts import models_v3 as wearmodels
from wearables.scripts import data_v3 as weardata
from wearables.scripts import eval_v3 as weareval
from wearables.scripts import utils as wearutils
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn import metrics as sklmetrics # non-standard
import numpy as np
import datetime
import pandas as pd

class InceptionTime_trainer():
    def __init__(self, exp, trial=0,
                 new_split=False,
                 model_path=None, out_file=None, target='GA',
                 lr=0.001, batch_size=32, n_epochs=2000, patience=100,
                 min_nb_epochs=500,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        '''
        TODO (ngr):
          - (enhancement) allow for multi-GPU training

        Arguments:
          exp (str): (optional) specify model and task with underscore, '_', in between
          trial (int): (optional, Default=0) 0 tends to indicate dev. Serve as input argument to re-run experiment
          patience (int or None): (optional, Default=100) if None, go through all epochs
        '''
        self.timer = wearutils.timer()
        self.timer.start() # time up to epoch
        self.out_file = out_file
        self.exp = '{}_{}'.format(exp, target)
        self.trial = trial
        self.new_split = new_split
        self.model_path = model_path
        self.target = target
        self.batch_size = batch_size
        self.patience = patience
        self.n_epochs = n_epochs
        self.min_nb_epochs = min_nb_epochs
        self.device = device
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        self.dataloaders = self.get_dataloaders()
        self.model = wearmodels.InceptionTime(1, self.out_dim, regression=True if self.tasktype=='regression' else False) 
        self.model = self.model.to(self.device)
        self.criterion = nn.MSELoss() if self.tasktype=='regression' else nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr,
                                          weight_decay=1e-4)

    def get_dataloaders(self, file='/home/ngr4/project/wearables/data/processed/model_data_210929.pkl'):
        '''Get train/val/test data.
        '''
        if self.new_split:
            data = weardata.load_actigraphy_md()
            data, md, _ = data['data'], data['ohmd'], data['md']
            model_data = weardata.get_Xy(data, md, pkl_out=os.path.join(self.model_path, 'data_{}_{}.pkl'.format(self.exp, self.trial)) if self.model_path is not None else None)
        else:
            model_data = weardata.load_datadict(fname=file)

        # select target
        X_train, y_train = model_data['X_train'], model_data['Y_train'].loc[:, self.target]
        X_val, y_val = model_data['X_val'], model_data['Y_val'].loc[:, self.target]
        X_test, y_test = model_data['X_test'], model_data['Y_test'].loc[:, self.target]

        self.tasktype = 'regression' if y_train.dtype == 'float64' else 'classification'
        self.out_dim = 1 # if self.tasktype=='regression' else len(target_id)

        data_train = weardata.actigraphy_dataset(X_train, y_train)
        data_val = weardata.actigraphy_dataset(X_val, y_val)
        data_test = weardata.actigraphy_dataset(X_test, y_test)

        dl_train = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        dl_val = torch.utils.data.DataLoader(data_val, batch_size=self.batch_size, shuffle=True)
        dl_test = torch.utils.data.DataLoader(data_test, batch_size=self.batch_size, shuffle=True)

        return {'train':dl_train, 'val':dl_val, 'test':dl_test,
                'md_train':model_data['Y_train'], 
                'md_val': model_data['Y_val'], 
                'md_test': model_data['Y_test']}

    def eval_performance(self, output, target, n_trials=10, nan20=False):
        # evals in numpy, so convert to host mem
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        if self.tasktype == 'regression':
            # Spearman's Rho
            rho, p = spearmanr(output, target)
            return rho.item()
        else:
            # AU-PRC vs. random (AU-PRC adjusted)
            if self.out_dim == 1:
                target = target.unsqueeze(1)
            auprc_model = weareval.auprc(output, target, nan20=nan20)
            random_clf = torch.distributions.normal.Normal(0, 1)
            auprc_random = 0.
            for n in range(n_trials):
                random_output = random_clf.sample((output.shape[0], output.shape[1]))
                if self.out_dim > 1:
                    random_output = torch.softmax(random_output, dim=-1)
                else:
                    random_output = torch.sigmoid(random_output)
                auprc_random += weareval.auprc(random_output, target, nan20=nan20)
            auprc_random = auprc_random / n_trials
            return (auprc_model - auprc_random) / (1 - auprc_random)

    def clear_modelpkls(self, best_epoch):
        files = glob.glob(os.path.join(self.model_path, '*-{}{}.pkl'.format(self.exp, self.trial)))
        for file in files:
            epoch_nb = int(os.path.split(file)[1].split('-{}{}.pkl'.format(self.exp, self.trial))[0])
            if epoch_nb != best_epoch:
                os.remove(file)

    def fit(self, verbose=True):
        '''Train model.
        '''
        # trackers
        total_loss = []
        total_loss_val = []
        total_eval = []
        total_eval_val = []
        bad_counter = 0
        best = np.inf
        best_epoch = 0

        print('\nStarting training after {:.0f}-s of setup...'.format(self.timer.stop()))
        for epoch in range(self.n_epochs):
            print(epoch)
            self.timer.start()
            self.model.train()

            # epoch trackers
            epoch_loss = []
            epoch_loss_val = []

            # train
            for i, batch in enumerate(self.dataloaders['train']):
                X_t, y, idx = batch['X'], batch['y'], batch['idx']
                X_t = X_t.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(X_t)
                if self.tasktype == 'regression':
                    output = output.squeeze()
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
                if i == 0: # bad idea for large training data 
                    output_total = output.detach()
                    target_total = y.detach()
                else:
                    output_total = torch.cat((output_total, output.detach()), dim=0)
                    target_total = torch.cat((target_total, y.detach()), dim=0)
            eval_metric = self.eval_performance(output_total, target_total)
            total_eval.append(eval_metric)

            # val
            self.model.eval()
            for i, batch in enumerate(self.dataloaders['val']):
                X_t, y, idx = batch['X'], batch['y'], batch['idx']
                X_t = X_t.to(self.device)
                y = y.to(self.device)

                output = self.model(X_t)
                if self.tasktype == 'regression':
                    output = output.squeeze()
                loss = self.criterion(output, y)
                epoch_loss_val.append(loss.item())
                if i == 0: # bad idea for large training data 
                    output_total = output.detach()
                    target_total = y.detach()
                else:
                    output_total = torch.cat((output_total, output.detach()), dim=0)
                    target_total = torch.cat((target_total, y.detach()), dim=0)
            eval_metric = self.eval_performance(output, y)
            total_eval_val.append(eval_metric)

            # track
            total_loss.append(np.mean(epoch_loss))
            total_loss_val.append(np.mean(epoch_loss_val))

            if verbose:
                print('Epoch {}\t<loss>={:.4f}\t<eval>={:.4f}\t<loss_val>={:.4f}\t<eval_val>={:.4f}\tin {:.0f}-s\telapsed: {:.1f}-min'.format(epoch,total_loss[-1], total_eval[-1], total_loss_val[-1], total_eval_val[-1], self.timer.stop(), self.timer.sum()/60))

            # save to model_zoo
            if total_loss_val[-1] < best:
                if self.model_path is not None:
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, '{}-{}{}.pkl'.format(epoch, self.exp, self.trial)))
                best = total_loss_val[-1]
                best_epoch = epoch
                self.clear_modelpkls(best_epoch)
                bad_counter = 0
            else:
                bad_counter += 1

            if self.patience is not None:
                if bad_counter == self.patience and epoch >= self.min_nb_epochs:
                    if self.model_path is not None:
                        self.clear_modelpkls(best_epoch)
                    break
            elif epoch==(self.n_epochs-1):
                if self.model_path is not None and bad_counter==0:
                    # save last one if  last is best, otherwise, keep best so far
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, '{}-{}{}.pkl'.format(epoch, self.exp, self.trial)))

        # track for later eval
        self.best_epoch = best_epoch
        self.total_loss = total_loss
        self.total_eval = total_eval
        self.total_loss_val = total_loss_val
        self.total_eval_val = total_eval_val
        
        print('\nOptimization finished!\tBest epoch: {}\tMax epoch: {}'.format(best_epoch, epoch))
        print('  exp: {}\ttrial: {}'.format(self.exp, self.trial))
        print('  training time elapsed: {}-h:m:s\n'.format(str(datetime.timedelta(seconds=self.timer.sum()))))

    def eval_test(self, modelpkl=None, eval_on_cpu=False, eval_trainset=False, verbose=True):
        '''Loads best model or existing one (from last epoch)

        NOTE: to trigger last epoch being used, also turn off patience

        Returns:
          (dict): with results dataframe and output with indexing for metadata
        '''
        if eval_on_cpu:
            device = torch.device('cpu') 
            self.model.to(device)
        else:
            device = self.device
        if self.model_path is not None and modelpkl is None:
            self.model.load_state_dict(torch.load(os.path.join(self.model_path, '{}-{}{}.pkl'.format(self.best_epoch, self.exp, self.trial)),
                                             map_location=device))
        elif modelpkl is not None:
            # load stored model
            self.model.load_state_dict(torch.load(modelpkl, map_location=device))
            # set these to None
            self.best_epoch = None
            self.total_loss = None
            self.total_eval = None
            self.total_loss_val = None
            self.total_eval_val = None
            
        # test
        self.model.eval()
        if eval_trainset:
            dataloader = self.dataloaders['train']
        else:
            dataloader = self.dataloaders['test']
        for i, batch in enumerate(dataloader):
            X_t, y, idx = batch['X'], batch['y'], batch['idx']
            X_t = X_t.to(device)
            y = y.to(device)

            output = self.model(X_t)
            if self.tasktype == 'regression':
                output = output.squeeze()
            if i==0:
                y_total = y.detach().cpu() 
                idx_total = idx.detach().cpu()
                yhat_total = output.detach().cpu()
            else:
                y_total = torch.cat((y_total, y.detach().cpu()), dim=0)
                idx_total = torch.cat((idx_total, idx.detach().cpu()), dim=0)
                yhat_total = torch.cat((yhat_total, output.detach().cpu()), dim=0)
        loss_test = self.criterion(yhat_total, y_total).item()
        eval_test = self.eval_performance(yhat_total, y_total)

        if eval_trainset:
            dataset = 'train'
        else:
            dataset = 'test'
        if verbose:
            print('{} set eval:'.format(dataset))
            print('  bst epoch: {}\n  <loss_{}>={:.4f}\n  eval_{}  ={:.4f}'.format(self.best_epoch, dataset,
                                                                                     loss_test, dataset, eval_test))

        # store to results file
        results_df = pd.DataFrame({'exp':self.exp,
                                   'trial':self.trial,
                                   'target':self.target,
                                   'tasktype':self.tasktype, 
                                   'eval_test':eval_test,
                                   'loss_test':loss_test,
                                   'bst_epoch':self.best_epoch,
                                   'loss_train':None,
                                   'eval_train':None,
                                   'loss_val':None,
                                   'eval_val':None,},
                                  index=[0])
        results_df.at[0, 'loss_train'] = self.total_loss
        results_df.at[0, 'eval_train'] = self.total_eval
        results_df.at[0, 'loss_val'] = self.total_loss_val
        results_df.at[0, 'eval_val'] = self.total_eval_val

        # save
        if self.out_file is not None:
            if os.path.exists(self.out_file):
                results_df.to_csv(self.out_file, mode='a', header=False)
            else:
                results_df.to_csv(self.out_file)
        
        # return metadata for convenience
        md = self.dataloaders['md_{}'.format(dataset)].reset_index()

        return {'md_idx':idx_total, 'y':y_total, 'yhat':yhat_total, 'results':results_df, 'md':md}


def chk_trainer():
    trainer = InceptionTime_trainer(model_path='/home/ngr4/scratch60',
                                             patience=None, n_epochs=1, min_nb_epochs=0,
                                             batch_size=32)
    trainer.fit()
    res = trainer.eval_test()
    return res


