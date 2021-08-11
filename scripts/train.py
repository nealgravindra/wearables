import os
import sys
sys.path.append('/home/ngr/gdrive/wearables/scripts/')
import models as wearmodels
import data as weardata
import utils as wearutils

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
import numpy as np

class InceptionTimeRegressor_trainer():
    def __init__(self, exp='InceptionTime_GA', trial=0,
                 model_path=None, out_file=None, target='GA',
                 lr=0.001, batch_size=32, n_epochs=500, patience=100,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        '''
        TODO (ngr):
          - (enhancement) allow for classification or regression baswed on target.
            Currently, only regression can be implemented.

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
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        self.dataloaders = self.get_dataloaders()
        self.model = wearmodels.InceptionTime(1, 1) # NOTE: change output to n_classes
        self.model = self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr,
                                          weight_decay=1e-3)



    def get_dataloaders(self, return_md=True):
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
        data = weardata.ppdata_frompkl()
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

    def eval(self, output, target, convert_to_numpy=False):
        if convert_to_numpy:
            output = output.detach().numpy()
            target = target.detach().numpy()
            return mean_absolute_error(target, output)
        else:
            return (output - target).abs().mean().item()

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
                eval_metric = self.eval(output.squeeze(), y)

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
                eval_metric = self.eval(output.squeeze(), y)

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
                bad_counter = 0
            else:
                bad_counter += 1

            if self.patience is not None:
                if bad_counter == self.patience:
                    if self.model_path is not None:
                        files = glob.glob(os.path.join(self.model_path, '*-{}{}.pkl'.format(self.exp, self.trial)))
                        for file in files:
                            epoch_nb = int(os.path.split(file)[1].split('-{}{}.pkl'.format(self.exp, self.trial))[0])
                            if epoch_nb != best_epoch:
                                os.remove(file)
                    break
            elif epoch==(n_epochs-1):
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
        print('  training time elapsed: {}-h:m:s\n'.format(str(datetime.timedelta(seconds=timer.sum()))))

    def eval_test(self):
        '''Loads best model or existing one (from last epoch)

        NOTE: to trigger last epoch being used, also turn off patience

        Returns:
          (dict): with results dataframe and output with indexing for metadata
        '''
        device = torch.device('cpu') # differs from self.device here
        self.model.to(device)
        if self.model_path is not None:
            self.model.load_state_dict(torch.load(os.path.join(self.model_path, '{}-{}{}.pkl'.format(self.best_epoch, self.exp, self.trial)),
                                             map_location=device))
        # test
        self.model.eval()
        for i, batch in enumerate(self.dataloaders['test']):
            X_t, y, idx = batch['X'], batch['y'], batch['idx']
            X_t = X_t.to(device)
            y = y.to(device)

            output = self.model(X_t)
            if i==0:
                y_total = y
                idx_total = idx
                yhat_total = output.squeeze()
            else:
                y_total = torch.cat((y_total, y), dim=0)
                idx_total = torch.cat((idx_total, idx), dim=0)
                yhat_total = torch.cat((yhat_total, output.squeeze()), dim=0)
        loss_test = self.criterion(yhat_total, y_total).item()
        eval_test = self.eval(yhat_total, y_total)

        print('Test set eval:')
        print('  bst epoch: {}\n  <loss_test>={:.4f}\n  acc_test   ={:.4f}'.format(self.best_epoch, loss_test, eval_test))

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

        return {'md_idx':idx_total, 'y':y_total, 'yhat':yhat_total, 'results':results_df}


def chk_trainer():
    import sys
    sys.path.append('/home/ngr/gdrive/wearables/scripts/')
    import train as weartrain
    trainer = weartrain.InceptionTimeRegressor_trainer(model_path='/home/ngr/gdrive/wearables/model_zoo',
                                                       patience=None, n_epochs=1,
                                                       batch_size=64)
    trainer.fit()
    res = trainer.eval_test()