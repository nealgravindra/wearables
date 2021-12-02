'''
Pytorch implementation of basic data augmentation for time series, 
  developed for input data, x, of shape (N, D, L) where 
  N = n samples, D = n dim, L = seq length

Basic ts augmentations are drawn from survey paper results, 
  REF: Iwana & Uchida, PLoS ONE, 2021 with code: 
  https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py
'''
import time
import warnings
import numpy as np
import random
import torch
import torch.nn.functional as F

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
    end = (target_len + start)
    # interpolation 
    xprime = torch.zeros_like(x)
    for i, ts in enumerate(x):
        xprime[i, :, :] = F.interpolate(x[i, :, start[i]:end[i]].unsqueeze(0), (x.shape[2]), mode='linear')
    return xprime
    
def window_warping(x, window_ratio=0.1, scales=[0.5, 2.]):
    warp_scale = torch.tensor(np.random.choice(scales, x.shape[0]), dtype=torch.float32)
    warp_size = int(window_ratio * x.shape[2])
    
    window_starts = torch.randint(low=1, high=x.shape[2] - warp_size - 1, size=(x.shape[0], ))
    window_ends = (window_starts + warp_size)
    
    # interpolation 
    xprime = torch.zeros_like(x)
    for i, ts in enumerate(x):
        start_seg = x[i, :, :window_starts[i]]
        window_seg = F.interpolate(x[i, :, window_starts[i]:window_ends[i]].unsqueeze(0), 
                                   scale_factor=warp_scale[i].item(), mode='linear').squeeze()
        end_seg = x[i, :, window_ends[i]:]
        warped = torch.cat((start_seg, window_seg, end_seg), dim=-1)
        xprime[i, :, :] = F.interpolate(warped.unsqueeze(0), size=(x.shape[2]), mode='linear')
    return xprime


def augment_data(data, mode=['random'], verbose=False):
    '''Applies transformations to mb data

    Arguments:
      mode (list) [optional, Default='random']: whether
        to apply randomly or a specific type or all in list.
        Options are one of ['random', 'all', 'scaling', 'slicing',
        'jitter', 'windowwarping', 'none'] or their abbreviations, ['random',
        'all', 'Sc', 'Sl', 'WW', 'J', 'N']. IF supplying random or all, only pass those.
      apply_per_epoch (bool) [optional, Default=False]: if apply per minibatch, shuffle list
        if apply per epoch, return list of and re-shuffle it at start of next epoch

      
    '''
    if verbose:
        print('Transforms requested: {}'.format(mode))
        tic = time.time()
    augapplied = []  
    transform_bank = ['N', 'Sc', 'Sl', 'WW', 'J']
    for i, augtype in enumerate(mode):
        if verbose:
            print('  applying {}'.format(augtype))
        if augtype == 'random':
            data, augapplied = augment_data(data, mode=[np.random.choice(transform_bank)])
            break
        elif augtype == 'all': # assume first pass
            random.shuffle(transform_bank)
            data, augapplied = augment_data(data, mode=transform_bank)
            break
        elif augtype == 'N':
            augapplied.append(augtype)
            continue
        elif augtype == 'Sc':
            augapplied.append(augtype)
            data = scaling(data)
        elif augtype == 'Sl':
            augapplied.append(augtype)
            data = slicing(data)
        elif augtype == 'WW':
            augapplied.append(augtype)
            data = window_warping(data)
        elif augtype == 'J':
            augapplied.append(augtype)
            data = jitter(data)
        else:
            warnings.warn('Augmentation requested but not implemented')
            print('No {} transformation available'.format(augtype))
    if verbose:
        print('  ... applied {} in {:.0f}-s'.format(augapplied, time.time() - tic))
    return data, augapplied
