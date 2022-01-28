'''
Description:
  House evaluation metrics and simulations to ensure that chosen metric 
    aligns with computational goals of the project.
'''
import os
import sys
import torch
import pickle
import sklearn.metrics as sklmetrics
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# metrics 
#   for reg: (MAE, MAPE [mean absolute percentage error], Spearman's rho, P_spearman) 
#   for clf: (adjusted AU-PRC, balanced acc adj)
def auprc(output, target, nan20=False):
    '''un-balanced macro-average
    '''
    precision = dict()
    recall = dict()
    metric = dict()
    for i in range(output.shape[1]):
        precision[i], recall[i], _ = sklmetrics.precision_recall_curve(target[:, i], output[:, i])
        metric[i] = sklmetrics.auc(recall[i], precision[i])
    if nan20:
        metric = {k:v if not np.isnan(v) else 0. for k,v in metric.items()}
    return np.nanmean([v for v in metric.values()])

def eval_output(output, target, tasktype='regression', n_trials=10, nan20=False):
    '''Evaluate the model output (logits) versus ground-truth.
    
    Arguments:
      output (torch.tensor OR np.ndarray): y_hat
      target (torch.tensor OR np.ndarray): y_true
    '''
    
    
    if tasktype == 'regression':
        # Spearman's Rho
        if not isinstance(output, np.ndarray):
            rho, p = spearmanr(output.numpy(), target.numpy())
            mae = (output - target).abs().mean().item()
            mape = ((output - target)/target).abs().mean().item()
        else: 
            rho, p = spearmanr(output, target)
            mae = np.mean(np.abs((output - target)))
            mape = np.mean(np.abs((output - target)/target))
        return {'mae': mae, 'mape': mape, 'rho': rho, 'P_rho': p}
    else:
        # AU-PRC vs. random (AU-PRC adjusted)
        if len(output.shape) == 1:
            if torch.is_tensor(target):
                target = target.unsqueeze(1)
                output = output.unsqueeze(1)
            else:
                target = np.expand_dims(target, 1)
                output = np.expand_dims(output, 1)
        auprc_model = auprc(output, target, nan20=nan20)
        random_clf = torch.distributions.Uniform(0, 1) # torch.distributions.normal.Normal(0, 1)
        auprc_random = 0.
        for n in range(n_trials):
            random_output = random_clf.sample((output.shape[0], output.shape[1]))
            if len(output.shape) > 1 and output.shape[1] > 1:
                random_output = torch.softmax(random_output, dim=-1)
            else:
                random_output = torch.sigmoid(random_output)
            auprc_random += auprc(random_output, target, nan20=nan20)
        auprc_random = auprc_random / n_trials
        auprc_adj = (auprc_model, auprc_random)
#         auprc_adj = (auprc_model - auprc_random) / (1 - auprc_random)

        # balanced acc 
        if len(output.shape) > 1:
            balanced_acc = []
            balanced_acc_adj = []
            for i in range(output.shape[1]):
                balanced_acc.append(sklmetrics.balanced_accuracy_score(target[:, i], output[:, i], adjusted=False))
                balanced_acc_adj.append(sklmetrics.balanced_accuracy_score(target[:, i], output[:, i], adjusted=True))
            balanced_acc = np.mean(balanced_acc)
            balanced_acc_adj = np.mean(balanced_acc_adj)
        else:
            balanced_acc = sklmetrics.balanced_accuracy_score(target, output, adjusted=False)
            balanced_acc_adj = sklmetrics.balanced_accuracy_score(target, output, adjusted=True)
        return {'auprc_model': auprc_model, 'auprc_adj': auprc_adj, 'balanced_acc': balanced_acc, 'balanced_acc_adj': balanced_acc_adj}


# eval DL models
class eval_trained():
    def __init__(self, trainer, modelpkl=None, split='test', 
                 two_outputs=False,
                 out_file=None):
        self.trainer = trainer
        self.exp_name = '{}_{}'.format(self.trainer.exp, self.trainer.trial)
        self.modelpkl = modelpkl
        self.split = split
        self.device = torch.device('cpu')
        self.trainer.model = self.trainer.model.to(self.device)
        self.trainer.model.eval()
        self.two_outputs = two_outputs # signal whether weights/attn/embedding also output
        self.get_model_output()
        self.eval_performance = eval_output(self.yhat, self.y, tasktype=trainer.data.tasktype)
        if out_file is not None:
            self.output_results(out_file)
    
    def get_model_output(self):
        self.results = pd.DataFrame()
        
        # data
        if self.split=='test': # switch for set to analyze
            dataloader = self.trainer.data.test_dl
        elif self.split=='train':
            dataloader = self.trainer.data.train_dl
        elif self.split=='val':
            dataloader = self.trainer.data.val_dl
        else:
            print('Must evaluate one of train/test/val split')
            
        # model
        if self.modelpkl is not None:
            self.trainer.model.load_state_dict(
                torch.load(self.modelpkl, map_location=self.device))
        self.trainer.model.eval()
        dataloader.num_workers = 1
        for i, batch in enumerate(dataloader):
            x, y, idx = batch['x'], batch['y'], batch['id']
            tic = time.time()

            if self.two_outputs:
                output, addl_out = self.trainer.model(x, addl_out=True)
            else:
                output = self.trainer.model(x)
            if self.trainer.data.tasktype == 'regression':
                output = output.squeeze()
            if i==0:
                y_total = y.detach()
                idx_total = idx
                yhat_total = output.detach()
                if self.two_outputs:
                    out2_total = addl_out.detach()
            else:
                y_total = torch.cat((y_total, y.detach()), dim=0)
                idx_total = idx_total + idx
                yhat_total = torch.cat((yhat_total, output.detach().reshape(-1, )), dim=0)
                if self.two_outputs:
                    out2_total = torch.cat((out2_total, addl_out.detach()), dim=0)            
                    
        # store
        self.y = y_total
        self.yhat = yhat_total
        self.id = idx_total
        if self.two_outputs:
            self.out2 = out2_total
        if 'MSELoss' in str(self.trainer.criterion.__class__) or 'NLLLoss' in str(self.trainer.criterion.__class__):
            self.loss_test = self.trainer.criterion(output, y).item()
        else:
            self.loss_test = self.trainer.criterion(output, y, self.trainer.model.parameters()).item()
        for i, k in enumerate(self.id):
            dt = pd.Series(self.trainer.data.data['data'][k]['md']).T
            dt['id'] = k
            dt['y'] = self.y[i].item()
            dt['yhat'] = self.yhat[i].item()
            self.results = self.results.append(dt, ignore_index=True)
            
    def output_results(self, file):
        dt = pd.DataFrame({'exp_trial':self.exp_name, 
                           'y':None, 'yhat':None, 
                           'loss':self.loss_test}, index=[0])
        dt.at[0, 'y'] = self.y
        dt.at[0, 'yhat'] = self.yhat
        for k in self.eval_performance.keys():
            dt[k] = self.eval_performance[k]
        
        if os.path.exists(file):
            dt.to_csv(file, mode='a', header=True)
        else:
            dt.to_csv(file)
            
def p_encoder(p):
    if p > 0.05:
        label = '' # n.s.
    elif p <= 0.001:
        label = '***'
    elif p <= 0.05 and p > 0.01:
        label = '*'
    elif p <= 0.01 and p > 0.001:
        label = '**'
    else: 
        label = 'Unclassified'
    return label

def summarize_long_table(results_longtab, metrics=['MAE', 'Rho'], group='Model', out_file=None):
    '''
    Arguments:
      results_longtab (pd.DataFrame): assumes that replicates are indicated in a trial col
        but otherwise repeated according to the group colname
    '''
    from scipy.stats import ttest_ind
    summary = pd.DataFrame()
    
    for g in results_longtab[group].unique():
        summary.loc[g, group] = g
        for m in metrics:
            temp = {}
            others = [gg for gg in results_longtab[group].unique() if gg!=g]
            a = results_longtab.loc[results_longtab[group]==g, m].to_numpy()
            summary.loc[g, m] = '{:.2f} ({:.2f})'.format(np.mean(a), np.std(a))
            for gg in others:
                b = results_longtab.loc[results_longtab[group]==gg, m].to_numpy()
                statistic, p = ttest_ind(a, b)
                temp['v.{}'.format(gg)] = (np.max(a) - np.max(b), p)
            # only retain min
            k2keep = min(temp, key=temp.get)
            summary.loc[g, 'Top-1 diff ({})'.format(m)] = '{:.2f} ({})'.format(temp[k2keep][0], k2keep)
            summary.loc[g, 'P ({})'.format(m)] = '{:.2e}{} ({})'.format(temp[k2keep][1], p_encoder(temp[k2keep][1]), k2keep)
    if out_file is not None:
        summary.to_csv(out_file)
    return summary
            
# ~20min for inference 
def merge_out2md(md, bst_trainerfp, bst_modelfp, return_embeds=True, out_file=None, verbose=False):
    def loadpkl(file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            f.close()
        return data
    trainer = loadpkl(bst_trainerfp)
    if verbose:
        total_t = time.time()
    dt = pd.DataFrame()
    if return_embeds:
        embeds = pd.DataFrame()
    for split in ['train', 'val', 'test']:
        if verbose:
            tic = time.time()
            print('Starting inference for {} set...'.format(split))

        evaluation = weareval.eval_trained(trainer, split=split,
                                           modelpkl=bst_modelfp,
                                           two_outputs=True)
        dt = dt.append(pd.DataFrame({
            'y':evaluation.y.numpy(), 'yhat':evaluation.yhat.numpy(), 
            'split':[split]*evaluation.y.shape[0],
            'error':(evaluation.yhat - evaluation.y).numpy()
        }, index=evaluation.id))
        if return_embeds:
            embeds = embeds.append(pd.DataFrame(evaluation.out2.numpy(), index=evaluation.id))
        if verbose:
            print('  inference for {} set done in {:.0f}-s\t{:.2f}-min elapsed'.format(split, time.time()-tic, (time.time()-total_t)/60))
    md = md.merge(dt, left_index=True, right_index=True, how='left')
    if out_file is not None:
        md.to_csv(out_file)
    if return_embeds:
        if out_file is not None:
            embeds.to_csv(os.path.join(os.path.split(out_file)[0], 'embeds_v52.csv'))
        return md, embeds
    else:
        return md
    
# PathExplain
import functools
import operator
import torch
from torch.autograd import grad
import numpy as np
from tqdm import *

def gather_nd(params, indices):
    """
    Args:
        params: Tensor to index
        indices: k-dimension tensor of integers. 
    Returns:
        output: 1-dimensional tensor of elements of ``params``, where
            output[i] = params[i][indices[i]]
            
            params   indices   output
            1 2       1 1       4
            3 4       2 0 ----> 5
            5 6       0 0       1
    """
    max_value = functools.reduce(operator.mul, list(params.size())) - 1
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i]*m
        m *= params.size(i)

    idx[idx < 0] = 0
    idx[idx > max_value] = 0
    return torch.take(params, idx)

class PathExplainerTorch(object):
    def __init__(self, model):
        self.model = model
        return
    
    def _get_ref_tensor(self,baseline,batch_size,num_samples):
        number_to_draw = num_samples * batch_size
        replace = baseline.shape[0] < number_to_draw
        sample_indices = np.random.choice(baseline.shape[0],
                                          size=number_to_draw,
                                          replace=replace)
        ref_tensor = baseline[sample_indices,:]
        
        return ref_tensor

    def _get_samples_input(self, input_tensor, baseline, 
                           num_samples, use_expectation):
        '''
        calculate interpolation points
        Args:
            input_tensor: Tensor of shape (batch, ...), where ... indicates
                          the input dimensions. 
            reference_tensor: A tensor of shape (batch, k, ...) where ... 
                indicates dimensions, and k represents the number of background 
                reference samples to draw per input in the batch.
        Returns: 
            samples_input: A tensor of shape (batch, k, ...) with the 
                interpolated points between input and ref.
            samples_delta: A tensor of shape (batch, 1, ...) with the
                difference between input and reference for each sample
        '''
        input_dims = list(input_tensor.size())[1:]
        num_input_dims = len(input_dims)
        batch_size = input_tensor.size()[0]
        
        if use_expectation:
            reference_tensor = self._get_ref_tensor(baseline,batch_size,num_samples)
            shape = reference_tensor.shape
            reference_tensor = reference_tensor.view(
                    batch_size, 
                    num_samples, 
                    *(shape[1:]))
            
            # Grab a [batch_size, k]-sized interpolation sample
            t_tensor = torch.FloatTensor(batch_size, num_samples).uniform_(0,1).to(reference_tensor.device)
            shape = [batch_size, num_samples] + [1] * num_input_dims
            interp_coef = t_tensor.view(*shape)

            # Evaluate the end points
            end_point_ref = (1.0 - interp_coef) * reference_tensor

            input_expand_mult = input_tensor.unsqueeze(1)
            end_point_input = interp_coef * input_expand_mult

            # Affine Combine
            samples_input = end_point_input + end_point_ref
            
        else:
            batch_size = input_tensor.size()[0]
            input_expand = input_tensor.unsqueeze(1)
            reps = np.ones(len(baseline.shape)).astype(int)
            reps[0] = batch_size
            reference_tensor = baseline.repeat(list(reps)).unsqueeze(1)
#             reference_tensor = torch.as_tensor(sampled_baseline).unsqueeze(1).to(baseline.device)
            scaled_inputs = [reference_tensor + (float(i)/(num_samples-1))*(input_expand - reference_tensor) \
                             for i in range(0,num_samples)]
            samples_input = torch.cat(scaled_inputs,dim=1)
        
        samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
        samples_delta = samples_delta.to(samples_input.device)
        
        return samples_input, samples_delta
    
    def _get_samples_delta(self, input_tensor, reference_tensor):
        input_expand_mult = input_tensor.unsqueeze(1)
        sd = input_expand_mult - reference_tensor
        return sd
    
    def _get_grads(self, samples_input, output_indices=None):

        grad_tensor = torch.zeros(samples_input.shape).float().to(samples_input.device)
        
        k_ = samples_input.shape[1]

        for i in range(k_):
            particular_slice = samples_input[:,i]
            batch_output = self.model(particular_slice)
            # should check that users pass in sparse labels
            # Only look at the user-specified label
            if batch_output.size(1) > 1:
                sample_indices = torch.arange(0,batch_output.size(0)).to(samples_input.device)
                indices_tensor = torch.cat([
                        sample_indices.unsqueeze(1), 
                        output_indices.unsqueeze(1)], dim=1)
                batch_output = gather_nd(batch_output, indices_tensor)

            model_grads = grad(
                    outputs=batch_output,
                    inputs=particular_slice,
                    grad_outputs=torch.ones_like(batch_output).to(samples_input.device),
                    create_graph=True)
            grad_tensor[:,i,:] = model_grads[0]
        return grad_tensor
           
    def attributions(self, input_tensor, baseline,
                     num_samples = 50, use_expectation=True, 
                     output_indices=None):
        """
        Calculate either Expected or Integrated Gradients approximation of 
        Aumann-Shapley values for the sample ``input_tensor``.
        Args:
            model (torch.nn.Module): Pytorch neural network model for which the
                output should be explained.
            input_tensor (torch.Tensor): Pytorch tensor representing the input
                to be explained.
            baseline (torch.Tensor): Pytorch tensor representing the baseline.
                If use_expectation is true, then baseline should be shape
                (num_refs, ...) where ... indicates the dimensionality
                of the input. Otherwise, baseline should be shape (1, ...).
            output_indices (optional, default=None): For multi-class prediction
        """
        equal_dims = baseline.shape[1:] == input_tensor.shape[1:]
        almost_equal_dims = baseline.shape == input_tensor.shape[1:]
        
        dev = input_tensor.device
        baseline = baseline.to(dev)
        
        input_tensor.requires_grad_ = True
        
        if use_expectation and not equal_dims:
            raise ValueError('baseline should be shape (num_refs, ...) \
                              where ... indicates the dimensionality   \
                              of the input')
            
        if not use_expectation and baseline.shape[0] != 1:
            if almost_equal_dims:
                baseline = baseline.unsqueeze(0)
            else:
                raise ValueError('baseline should be shape (...)           \
                                  where ... indicates the dimensionality   \
                                  of the input')
        
        samples_input, samples_delta = self._get_samples_input(input_tensor, baseline,
                                                               num_samples, use_expectation)
        grad_tensor = self._get_grads(samples_input, output_indices)
        mult_grads = samples_delta * grad_tensor
        attributions = mult_grads.mean(1)
        
        return attributions
    
    def interactions(self, input_tensor, baseline,
                     num_samples=50, use_expectation=True,
                     output_indices=None, interaction_index=None,
                     verbose=True):
        """
            input_tensor (torch.Tensor): Pytorch tensor representing the input
                to be explained.
            baseline (torch.Tensor): Pytorch tensor representing the baseline.
                If use_expectation is true, then baseline should be shape
                (num_refs, ...) where ... indicates the dimensionality
                of the input. Otherwise, baseline should be shape (1, ...)
            num_samples: The number of samples to use when computing the
                         expectation or integral.
            use_expectation: If True, this samples baselines and interpolation
                             constants uniformly at random (expected gradients).
                             If False, then this assumes num_refs=1 in which
                             case it uses the same baseline for all inputs,
                             or num_refs=batch_size, in which case it uses
                             baseline[i] for inputs[i] and takes 100 linearly spaced
                             points between baseline and input (integrated gradients).
            output_indices:  If this is None, then this function returns the
                             attributions for each output class. This is rarely
                             what you want for classification tasks. Pass an
                             integer tensor of shape [batch_size] to
                             index the output output_indices[i] for
                             the input inputs[i].
            interaction_index: Either None or an index into the input. If the latter,
                               will compute the interactions with respect to that
                               feature. This parameter should index into a batch
                               of inputs as inputs[(slice(None) + interaction_index)].
                               For example, if you had images of shape (32, 32, 3)
                               and you wanted interactions with respect
                               to pixel (i, j, c), you should pass
                               interaction_index=[i, j, c].
        
        """
        
        if len(input_tensor.shape) != 2:
            raise ValueError('PyTorch Explainer only supports ' + \
                             'interaction for 2D input tensors!')
        
        equal_dims = baseline.shape[1:] == input_tensor.shape[1:]
        almost_equal_dims = baseline.shape == input_tensor.shape[1:]
        
        if use_expectation and not equal_dims:
            raise ValueError('baseline should be shape (num_refs, ...) \
                              where ... indicates the dimensionality   \
                              of the input')
            
        if not use_expectation and baseline.shape[0] != 1:
            if almost_equal_dims:
                baseline = baseline.unsqueeze(0)
            else:
                raise ValueError('baseline should be shape (...)           \
                                  where ... indicates the dimensionality   \
                                  of the input')
        
        inner_loop_nsamples = int(round(np.sqrt(num_samples)))
        
        samples_input, samples_delta = self._get_samples_input(input_tensor, baseline,
                                                               inner_loop_nsamples, use_expectation)
        
        if interaction_index is not None:
            interaction_mult_tensor = torch.zeros([input_tensor.size(0), samples_input.size(1), input_tensor.size(1)])
        else:
            interaction_mult_tensor = torch.zeros([input_tensor.size(0), samples_input.size(1), 
                                                   input_tensor.size(1), input_tensor.size(1)])
            
        ig_tensor = torch.zeros(samples_input.shape).float()
        
        loop_num = inner_loop_nsamples
        
        if verbose:
            iterable = tqdm(range(loop_num))
        else:
            iterable = range(loop_num)
        
        for i in iterable:
            
            particular_slice = samples_input[:,i]
            ig_tensor[:,i,:] = self.attributions(particular_slice, baseline,
                                                 num_samples=inner_loop_nsamples, use_expectation=use_expectation,
                                                 output_indices=output_indices)
            
            if interaction_index is not None:
                second_grads = grad(
                        outputs=ig_tensor[:,i,interaction_index],
                        inputs=particular_slice,
                        grad_outputs=torch.ones_like(ig_tensor[:,i,interaction_index]),
                        create_graph=True)[0].detach()
                interaction_mult_tensor[:,i,:] = second_grads

            else:
                for feature in range(input_tensor.size(1)):
                    second_grads = grad(
                        outputs=ig_tensor[:,i,feature],
                        inputs=particular_slice,
                        grad_outputs=torch.ones_like(ig_tensor[:,i,feature]),
                        create_graph=True)[0].detach()
                    interaction_mult_tensor[:,i,feature,:] = second_grads

        interaction_mult_tensor = interaction_mult_tensor.to(samples_delta.device)
        if interaction_index is not None:
            interaction_tensor = interaction_mult_tensor * samples_delta
        else:
            interaction_tensor = interaction_mult_tensor * samples_delta.unsqueeze(2)
        interactions = interaction_tensor.mean(1)
        
        return interactions
    
    
def featattr_peruid(X, md, uids, explainer, trainer):
    '''
    Returns:
      df (pd.DataFrame)
      uids (dict): 'groupname': list of unique ids
    '''
    total_t = time.time()
    df = pd.DataFrame()
    counter = 0
    for i, g in enumerate(uids.keys()):
        for uid in uids[g]:
            idx = np.where(md.index==uid)[0].item()
            x = X[idx, :, :].unsqueeze(0)
            x.requires_grad = True
            print('starting idx: {}, unique id: {}\{:.2f}-min elapsed'.format(idx, uid, (time.time()-total_t)/60))
            if counter==0: # only need to call once
                x_baseline = torch.zeros_like(x)
                x_baseline.requires_grad = True
            attr = explainer.attributions(input_tensor=x,
                                          baseline=x_baseline,
                                          num_samples=20,
                                          use_expectation=True,
                                          output_indices=0)
            dt = pd.DataFrame(attr[0, 0, :].detach().numpy(), columns=['attr'])
            dt['uid'] = uid
            dt['sleep'] = trainer.data.data['data'][uid]['sleep'][:-1].to_numpy()
            dt['group'] = g
            df = df.append(dt)
            counter += 1
    return df


# if __name__ == '__main__':
#     mode = 'weighted_AU-PRC'
#     nan20 = False
#     p_class=[
#             [0.5, 0.4, 0.1],
#             [0.5, 0.49, 0.01],
#             [0.5, 0.499, 0.001],
#             [0.8, 0.1, 0.1],
#             [0.9, 0.09, 0.01],
#             [0.9, 0.099, 0.001]
#             ]
#     # p_class = [[0.5], [0.6], [0.7], [0.8], [0.9], [99/100], [999/1000]]
#     res_multi = eval_evalmetric(p_class=p_class).experiment(verbose=False, mode=mode, nan20=nan20)
#     fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#     p = sns.boxplot(x='P_class', y=mode, data=res_multi, ax=ax)
#     p.set_xticklabels(p.get_xticklabels(), rotation=45)
#     fig.tight_layout()
#     plt.show()

# statistical tests
def pd_chisq(df, feat, groupby='Error group'):
    from scipy.stats import chi2_contingency
    obs = md.groupby([groupby, feat]).size().unstack(fill_value=0)
    chi2, p, dof, expected = chi2_contingency(obs)
    return p, obs

def pd_kruskalwallis(df, feat, groupby='Error group'):
    size = []
    for i, g in enumerate(df[groupby].unique()):
        dt = df.loc[df[groupby]==g, feat].to_numpy()
        size.append(dt.shape[0])
        if i==0:
            X = dt
        else:
            X = np.concatenate((X, dt))
    X = np.split(X, np.cumsum(size[:-1]))
    from scipy.stats import kruskal
    statistic, p = kruskal(*X)
    return p

def pd_purity():
    '''Silhouette coefficient per var per cluster'''
    raise NotImplemented