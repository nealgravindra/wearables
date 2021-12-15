import sys
sfp = '/home/ngr4/project/'
sys.path.append(sfp)
from wearables.scripts import model as wearmodels
from wearables.scripts import data_v42 as weardata
from wearables.scripts import eval_v42 as weareval
from wearables.scripts import utils as wearutils
from wearables.scripts import data_augmentation as aug

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import pickle
import numpy as np
import datetime
import pandas as pd
import glob

# objectives
class MSEL1(nn.Module):
    def __init__(self, lambda_l1=0.001):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.MSE = nn.MSELoss()
        
    def L1(self, params):
        return sum(p.abs().sum() for p in params)
    
    def forward(self, output, target, params):
        return self.MSE(output, target) + self.lambda_l1*self.L1(params)

class train():
    '''
    Arguments:
      hyperparameters *dict*: contains values for any of 'lr', 'criterion',
        'lambda_l2', 'batch_size'
    '''
    def __init__(self, 
                 model, 
                 exp=None, 
                 target='GA', 
                 trial=0, 
                 load_splits=None, 
                 model_path=None, 
                 out_file=None, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 **hyperparams):
        '''
        TODO:
          1. saved split of data loader only has specified target (GA)
        '''

        self.timer = wearutils.timer()
        self.timer.start() # time up to epoch
        if 'batch_size' not in hyperparams.keys():
            hyperparams['batch_size'] = 32
        if 'lr' not in hyperparams.keys():
            hyperparams['lr'] = 0.001
        if 'lambda_l2' not in hyperparams.keys():
            hyperparams['lambda_l2'] = 5e-4
        if 'nb_epochs' not in hyperparams.keys():
            hyperparams['nb_epochs'] = 10000
        if 'patience' not in hyperparams.keys():
            hyperparams['patience'] = 2500
        if 'criterion' not in hyperparams.keys():
            hyperparams['criterion'] = nn.MSELoss()
        if 'min_nb_epochs' not in hyperparams.keys():
            hyperparams['min_nb_epochs'] = 2000
        if 'shuffle_label' not in hyperparams.keys():
            hyperparams['shuffle_label'] = False
        if 'aug_mode' not in hyperparams.keys():
            hyperparams['aug_mode'] = ['random'] 
        if 'aug_per_epoch' not in hyperparams.keys():
            hyperparams['aug_per_epoch'] = False
        self.hyperparams = hyperparams
        
        print('  note: running with following specifications: ', hyperparams)
        self.out_file = out_file
        if exp is None:
            exp = str(model.__class__).split('.')[-1][:-2] # model name
        self.exp = '{}_{}'.format(exp, target)
        self.trial = trial
        self.load_splits = load_splits
        self.model_path = model_path
        self.target = target
        self.batch_size = self.hyperparams['batch_size']
        self.patience = self.hyperparams['patience']
        self.nb_epochs = self.hyperparams['nb_epochs']
        self.min_nb_epochs = self.hyperparams['min_nb_epochs']
        self.shuffle_label = self.hyperparams['shuffle_label']
        self.aug_mode = self.hyperparams['aug_mode']
        self.aug_per_epoch = self.hyperparams['aug_per_epoch']
        self.device = device
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # data
        self.get_dataloaders()
        self.model = model.to(self.device)
        self.criterion = hyperparams['criterion']#MSEL1(lambda_l1=hyperparams['lambda_l1'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hyperparams['lr'],
            weight_decay=self.hyperparams['lambda_l2'])

    def get_dataloaders(self):
        '''Get train/val/test data.
        '''
        if self.load_splits is None:
            self.data = weardata.torch_dataloaders(
                self.target, 
                batch_size=self.batch_size)
        else:
            with open(self.load_splits, 'rb') as f:
                self.data = pickle.load(f)
                f.close()

    def clear_modelpkls(self, best_epoch):
        files = glob.glob(os.path.join(self.model_path, '*-{}{}.pkl'.format(self.exp, self.trial)))
        for file in files:
            epoch_nb = int(os.path.split(file)[1].split('-{}{}.pkl'.format(self.exp, self.trial))[0])
            if epoch_nb != best_epoch:
                os.remove(file)

    def fit(self, verbose=True):
        
        # trackers
        total_loss = []
        total_loss_val = []
        bad_counter = 0
        best = np.inf
        best_epoch = 0

        print('\nStarting training after {:.0f}-s of setup...'.format(self.timer.stop()))
        for epoch in range(self.nb_epochs):
            self.timer.start()
            self.model.train()

            # epoch trackers
            epoch_loss = []
            epoch_loss_val = []

            # train
            for i, batch in enumerate(self.data.train_dl):
                x, y, idx = batch['x'], batch['y'], batch['id']
                if self.aug_mode is not None:
                    if not self.aug_per_epoch:
                        x, _ = aug.augment_data(x, mode=self.aug_mode)
                    else:
                        if i == 0:
                            x, augsapplied = aug.augment_data(x, mode=self.aug_mode)
                        else:
                            x, _ = aug.augment_data(x, mode=augsapplied)
                if self.shuffle_label:
                    y = y[torch.randperm(y.shape[0])] 
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                if self.data.tasktype == 'regression':
                    output = output.squeeze()
                if 'MSELoss' in str(self.criterion.__class__) or 'NLLLoss' in str(self.criterion.__class__):
                    loss = self.criterion(output, y)
                else:
                    loss = self.criterion(output, y, self.model.parameters())
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())


            # val
            self.model.eval()
            for i, batch in enumerate(self.data.val_dl):
                x, y, idx = batch['x'], batch['y'], batch['id']
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                if self.data.tasktype == 'regression':
                    output = output.squeeze()
                if 'MSELoss' in str(self.criterion.__class__) or 'NLLLoss' in str(self.criterion.__class__):
                    loss = self.criterion(output, y)
                else:
                    loss = self.criterion(output, y, self.model.parameters())
                epoch_loss_val.append(loss.item())

            # track
            total_loss.append(np.mean(epoch_loss))
            total_loss_val.append(np.mean(epoch_loss_val))

            if verbose:
                print('Epoch {}\t<loss>={:.4f}\t<loss_val>={:.4f}\tin {:.0f}-s\telapsed: {:.1f}-min'.format(
                    epoch, total_loss[-1], total_loss_val[-1], self.timer.stop(), self.timer.sum()/60))

            # save to model_zoo
            if total_loss_val[-1] < best and epoch > self.min_nb_epochs:
                best = total_loss_val[-1]
                best_epoch = epoch
                bad_counter = 0
                if self.model_path is not None:
                    torch.save(self.model.state_dict(), 
                               os.path.join(self.model_path, 
                                            '{}-{}{}.pkl'.format(best_epoch, self.exp, self.trial)))
                    self.clear_modelpkls(best_epoch)
            else:
                bad_counter += 1

            if self.patience is not None:
                if bad_counter == self.patience and best_epoch >= self.min_nb_epochs:
                    if self.model_path is not None:
                        self.clear_modelpkls(best_epoch)
                    break

        # track for later eval
        self.best_epoch = best_epoch
        self.loss = total_loss
        self.loss_val = total_loss_val
        
        # store to results file
        results_df = pd.DataFrame({'exp':self.exp,
                                   'trial':self.trial,
                                   'target':self.target,
                                   'hyperparams': None,
                                   'tasktype':self.data.tasktype, 
                                   'bst_epoch':self.best_epoch,
                                   'loss_train':None,
                                   'loss_val':None,},
                                  index=[0])
        results_df.at[0, 'hyperparams'] = self.hyperparams
        results_df.at[0, 'loss_train'] = self.loss
        results_df.at[0, 'loss_val'] = self.loss_val
        self.training_summary = results_df

        # save
        if self.out_file is not None:
            if os.path.exists(self.out_file):
                results_df.to_csv(self.out_file, mode='a', header=False)
            else:
                results_df.to_csv(self.out_file)
                
            # save trainer
            if self.model_path is not None:    
                # output to model zoo
                with open(os.path.join(self.model_path, 
                                       'trainer_{}{}.pkl'.format(self.exp, self.trial)), 'wb') as f:
                    pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print('\nOptimization finished!\tBest epoch: {}\tMax epoch: {}'.format(best_epoch, epoch))
        print('  exp: {}\ttrial: {}'.format(self.exp, self.trial))
        print('  training time elapsed: {}-h:m:s\n'.format(str(datetime.timedelta(seconds=self.timer.sum()))))





