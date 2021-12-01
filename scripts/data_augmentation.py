'''
Pytorch implementation of basic data augmentation for time series, 
  developed for input data, x, of shape (N, D, L) where 
  N = n samples, D = n dim, L = seq length

Basic ts augmentations are drawn from survey paper results, 
  REF: Iwana & Uchida, PLoS ONE, 2021 with code: 
  https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py
'''

import torch
import numpy as np

def augment_data(data_input, mode='random'):
    '''Applies transformations to mb data

    Arguments:
      mode (str) [optional, Default='random']: whether
        to apply randomly or a specific type or all in bank.
        Options are one of ['random', 'all', 'scaling', 'slicing',
        'jitter', 'windowwarping', 'none'] or their abbreviations, ['random',
        'all', 'Sc', 'Sl', 'WW', 'J', 'N']

      
    '''
    if mode.lower() == 'random':
        print('not implemented')
    elif mode.lower() == 'all':
        print('do all')

    return data_output

def jitter(x, mu=0.0, sigma=0.03):
    return x + torch.normal(mu, sigma, size=x.size())

def scaling(x, sigma=0.2):
    '''
    Arguments:
      x (torch.tensor, dtype=torch.float32): shape = (N, D, L) where 
        N = batch size, D = num dimensions, L = seq length
    '''
    factor = torch.normal(1.0, sigma, size=(x.shape[0], 1))
    factor = factor.repeat(1, x.shape[1]).unsqueeze(-1)
    return factor * x


def slicing(x, target_ratio=0.9, target_len=None):
    '''
    Arguments:
      x (torch.tensor, dtype=torch.float32): shape=(N, D, L)
      target_ratio (float) [optional, Default=0.9]: range (0, 1) to set target len 
      target_len (None or float) [ooptional, Default=None] if float, then manually set,
        and ignore target_ratio
    '''
    if target_len is None:
        target_len = int(target_ratio * x.shape[2])
    else:
        target_ratio = x.shape[2] / target_len
    start = torch.randint(high=(x.shape[2] - target_len), size=(x.shape[0], ))
    end = (target_len + strt)
    # interpolation 
    xprime = torch.zeros_like(x)
    for i, ts in enumerate(x):
        for dim in range(x.shape[1]):
            xprime[i, dim, :] = 
    return xprime
    
    # old    
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

# NOT EDITED
def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret
