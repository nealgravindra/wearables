import os
import sys
sfp = '/home/ngr4/project/wearables/scripts/' # '/home/ngr/gdrive/wearables/scripts/' OR '/home/ngr4/project/wearables/scripts/'
sys.path.append(sfp)
import models as wearmodels
import data as weardata
import utils as wearutils
import glob

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import average_precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import numpy as np
import datetime
import pandas as pd

class InceptionTimeRegressor_trainer():
    def __init__(self, exp='InceptionTime_GA', trial=0,
                 model_path=None, out_file=None, target='GA',
                 lr=0.001, batch_size=32, n_epochs=500, patience=100,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 use_multi_gpus=False):
        '''DEPRECATED for versions beyond v0.2
        
        Arguments:
          exp (str): (optional) specify model and task with underscore, '_', in between
          trial (int): (optional, Default=0) 0 tends to indicate dev. Serve as input argument to re-run experiment
          patience (int or None): (optional, Default=100) if None, go through all epochs
        '''
        self.timer = wearutils.timer()
        self.timer.start() # time up to epoch
        self.out_file = out_file
        self.exp = exp
        self.trial = trial
        self.model_path = model_path
        self.target = target
        self.batch_size = batch_size
        self.patience = patience
        self.n_epochs = n_epochs
        self.device = device
        self.use_multi_gpus = use_multi_gpus
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        self.dataloaders = self.get_dataloaders()
        self.model = wearmodels.InceptionTime(1, 1) # NOTE: change output to n_classes
        if self.use_multi_gpus:
            selfmodel = nn.DataParallel(self.model) # DEPCREATED, see: https://pytorch.org/docs/master/notes/cuda.html#best-practices
#             torch.distributed.init_process_group(backend='nccl', world_size=torch.cuda.device_count())
#             self.model = nn.parallel.DistributedDataParallel(self.model)
        self.model = self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr,
                                          weight_decay=1e-3)



    def get_dataloaders(self, return_md=True,
                        file='/home/ngr4/project/wearables/data/processed/datapkl_Xactigraphy_Ymd_trainvaltest210803.pkl'):
        '''Get train/val/test data.

        TODO (ngr):
          - (enhancement): flexibly allow for target_id to be used to specify
            classification or regression and adjust optimizer accordingly

        Arguments:
          return_md (bool): (optional, Default=False) return the metadata (as dict)
            for the samples.

        Requirements:
          - need file to be specified in weardata.ppdata_frompkl(file) where
            file='*/datapkl_Xactigraphy_Ymd_trainvaltest210803.pkl'
        '''
        data = weardata.ppdata_frompkl(file=file)
        y_dict, target_id = weardata.md2y({k:data[k] for k in data.keys() if 'X' not in k}, label=self.target, wide=True)

        data_train = weardata.actigraphy_dataset(data['X_train'], y_dict['Y_train'])
        data_val = weardata.actigraphy_dataset(data['X_val'], y_dict['Y_val'])
        data_test = weardata.actigraphy_dataset(data['X_test'], y_dict['Y_test'])

        dl_train = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        dl_val = torch.utils.data.DataLoader(data_val, batch_size=self.batch_size, shuffle=True)
        dl_test = torch.utils.data.DataLoader(data_test, batch_size=self.batch_size, shuffle=True)

        if return_md:
            return {'train':dl_train, 'val':dl_val, 'test':dl_test,
                    'md_train':data['Y_train'], 'md_val':data['Y_val'], 'md_test':data['Y_test']}
        else:
            return {'train':dl_train, 'val':dl_val, 'test':dl_test}

    def performance_eval(self, output, target, convert_to_numpy=False):
        if convert_to_numpy:
            output = output.detach().numpy()
            target = target.detach().numpy()
            return mean_absolute_error(target, output)
        else:
            return (output - target).abs().mean().item() 
        
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
            epoch_eval = []
            epoch_eval_val = []

            # train
            for i, batch in enumerate(self.dataloaders['train']):
                X_t, y, idx = batch['X'], batch['y'], batch['idx']
                X_t = X_t.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(X_t)
                loss = self.criterion(output.squeeze(), y)
                loss.backward()
                self.optimizer.step()
                eval_metric = self.performance_eval(output.squeeze(), y)

                # track (change per set)
                epoch_loss.append(loss.item())
                epoch_eval.append(eval_metric)

            # val
            self.model.eval()
            for i, batch in enumerate(self.dataloaders['val']):
                X_t, y, idx = batch['X'], batch['y'], batch['idx']
                X_t = X_t.to(self.device)
                y = y.to(self.device)

                output = self.model(X_t)
                loss = self.criterion(output.squeeze(), y)
                eval_metric = self.performance_eval(output.squeeze(), y)

                # track (change per set)
                epoch_loss_val.append(loss.item())
                epoch_eval_val.append(eval_metric)

            # track
            total_loss.append(np.mean(epoch_loss))
            total_eval.append(np.mean(epoch_eval))
            total_loss_val.append(np.mean(epoch_loss_val))
            total_eval_val.append(np.mean(epoch_eval_val))

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
                if bad_counter == self.patience:
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

    def eval_test(self, modelpkl=None, eval_on_cpu=False, eval_trainset=False):
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
            if i==0:
                y_total = y.detach().cpu() 
                idx_total = idx.detach().cpu()
                yhat_total = output.squeeze().detach().cpu()
            else:
                y_total = torch.cat((y_total, y.detach().cpu()), dim=0)
                idx_total = torch.cat((idx_total, idx.detach().cpu()), dim=0)
                yhat_total = torch.cat((yhat_total, output.squeeze().detach().cpu()), dim=0)
        loss_test = self.criterion(yhat_total, y_total).item()
        eval_test = self.performance_eval(yhat_total, y_total)

        if eval_trainset:
            dataset = 'train'
        else:
            dataset = 'test'
        print('{} set eval:'.format(dataset))
        print('  bst epoch: {}\n  <loss_{}>={:.4f}\n  eval_{}  ={:.4f}'.format(self.best_epoch, dataset,
                                                                                 loss_test, dataset, eval_test))

        # store to results file
        results_df = pd.DataFrame({'exp':self.exp,
                                   'trial':self.trial,
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
        
        # merge metadata
        md_merged = self.dataloaders['md_{}'.format(dataset)].reset_index().merge(
            pd.DataFrame({'Absolute Error':(yhat_total-y_total).abs().numpy(),
                          'yGA':y_total.numpy(), 
                          'yhatGA':yhat_total.numpy()}, 
                         index=idx_total.numpy()), left_index=True, right_index=True)

        return {'md_idx':idx_total, 'y':y_total, 'yhat':yhat_total, 'results':results_df, 'md':md_merged}
    
class InceptionTime_trainer():
    def __init__(self, exp='InceptionTimev0.2', trial=0,
                 model_path=None, out_file=None, target='GA',
                 lr=0.001, batch_size=32, n_epochs=2000, patience=100,
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
        self.model_path = model_path
        self.target = target
        self.batch_size = batch_size
        self.patience = patience
        self.n_epochs = n_epochs
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



    def get_dataloaders(self, return_md=True,
                        file='/home/ngr4/project/wearables/data/processed/datapkl_Xactigraphy_Ymd_trainvaltest210803.pkl'):
        '''Get train/val/test data.

        Arguments:
          return_md (bool): (optional, Default=False) return the metadata (as dict)
            for the samples.

        Requirements:
          - need file to be specified in weardata.ppdata_frompkl(file) where
            file='*/datapkl_Xactigraphy_Ymd_trainvaltest210803.pkl'
        '''
        data = weardata.ppdata_frompkl(file=file)
        y_dict, target_id = weardata.md2y({k:data[k] for k in data.keys() if 'X' not in k}, label=self.target, wide=True)
        self.tasktype = 'regression' if isinstance(target_id, str) else 'classification'
        self.out_dim = 1 if self.tasktype=='regression' else len(target_id)
        self.target_id = target_id

        data_train = weardata.actigraphy_dataset(data['X_train'], y_dict['Y_train'])
        data_val = weardata.actigraphy_dataset(data['X_val'], y_dict['Y_val'])
        data_test = weardata.actigraphy_dataset(data['X_test'], y_dict['Y_test'])

        dl_train = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        dl_val = torch.utils.data.DataLoader(data_val, batch_size=self.batch_size, shuffle=True)
        dl_test = torch.utils.data.DataLoader(data_test, batch_size=self.batch_size, shuffle=True)

        if return_md:
            return {'train':dl_train, 'val':dl_val, 'test':dl_test,
                    'md_train':data['Y_train'], 'md_val':data['Y_val'], 'md_test':data['Y_test']}
        else:
            return {'train':dl_train, 'val':dl_val, 'test':dl_test}

    def eval_performance(self, output, target, convert_to_numpy=False, eval_type=None):
        if self.tasktype == 'regression' and eval_type is None:
            # MAE
            return (output - target).abs().mean().item()
        elif self.tasktype != 'regression' and eval_type is None:
            # accuracy
            preds = output.exp().max(1)[1].type_as(target)
            correct = preds.eq(target).double()
            correct = correct.sum()
            return (correct / len(target)).item()
        elif self.tasktype == 'regression' and eval_type.split('_')[1]=='R2':
            return r2_score(target, output)
        elif self.tasktype != 'regression' and eval_type.split('_')[0]=='AP':
            y_true = torch.zeros(target.shape[0], self.out_dim)
            y_true[torch.arange(target.shape[0]), target] = 1.
            aps = []
            output = output.exp()
            for i in range(self.out_dim):
                aps.append(average_precision_score(y_true[:, i], output[:, i], average='micro'))
            return np.nanmean(aps)
        elif self.tasktype != 'regression' and eval_type.split('_')[0]=='AUPRC':
            # define positive class as minority class, as in val set, similar to testing (majority are neg., want to be precise and sensitive when get a signal)
            
        elif self.tasktype == 'regression' and eval_type.split('_')[1]=='ExplainedVar':
            return explained_variance_score(target, output)
        else:
            import warnings
            warnings.warn('Valid evaluation metric was not specified.')
            return None
            
            
        
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
            epoch_eval = []
            epoch_eval_val = []

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
                eval_metric = self.eval_performance(output, y)

                # track (change per set)
                epoch_loss.append(loss.item())
                epoch_eval.append(eval_metric)

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
                eval_metric = self.eval_performance(output, y)

                # track (change per set)
                epoch_loss_val.append(loss.item())
                epoch_eval_val.append(eval_metric)

            # track
            total_loss.append(np.mean(epoch_loss))
            total_eval.append(np.mean(epoch_eval))
            total_loss_val.append(np.mean(epoch_loss_val))
            total_eval_val.append(np.mean(epoch_eval_val))

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
                if bad_counter == self.patience:
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

    def eval_test(self, modelpkl=None, eval_on_cpu=False, eval_trainset=False, eval_type='AP_R2', verbose=True):
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
                otuput = output.squeeze()
            if i==0:
                y_total = y.detach().cpu() 
                idx_total = idx.detach().cpu()
                yhat_total = output.detach().cpu()
            else:
                y_total = torch.cat((y_total, y.detach().cpu()), dim=0)
                idx_total = torch.cat((idx_total, idx.detach().cpu()), dim=0)
                yhat_total = torch.cat((yhat_total, output.detach().cpu()), dim=0)
        loss_test = self.criterion(yhat_total, y_total).item()
        eval_test = self.eval_performance(yhat_total, y_total, eval_type=eval_type)

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
                                   'eval_val':None,
                                   'target_id':None},
                                  index=[0])
        results_df.at[0, 'loss_train'] = self.total_loss
        results_df.at[0, 'eval_train'] = self.total_eval
        results_df.at[0, 'loss_val'] = self.total_loss_val
        results_df.at[0, 'eval_val'] = self.total_eval_val
        results_df.at[0, 'target_id'] = self.target_id

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
    trainer = InceptionTimeRegressor_trainer(model_path='/home/ngr4/project/wearables/model_zoo',
                                             patience=None, n_epochs=1,
                                             batch_size=32)
    trainer.fit()
    res = trainer.eval_test()
    return res


