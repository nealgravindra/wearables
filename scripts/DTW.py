'''
Script houses all aspects of analysis using
  DTW as a distance metric, including custom kNN 
  classification and regression tools to circumvent 
  sklearn slowness.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import time
import pickle
import datetime
import re

import sys
sys.path.append('/home/ngrav/project/')
from wearables.scripts import utils as wearutils
from wearables.scripts import data_v42 as weardata
from wearables.scripts import train_v42 as weartrain
from wearables.scripts import eval_v42 as weareval
from wearables.scripts import model as wearmodels

import torch
import torch.nn as nn
import torch.nn.functional as F

plt.rc('font', size = 9)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['legend.frameon']=False
plt.rcParams['axes.grid']=False
# plt.rcParams['legend.markerscale']=0.5
plt.rcParams['savefig.dpi'] = 600
sns.set_style("ticks")

from scipy.spatial.distance import pdist, squareform
import fastdtw
import umap
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg as la

def data_from_trainer(trainer_fp, split='train'):
    '''Save the trainer, extract data from it with metadata.
    
    Arguments:
      trainer_fp (str) filepath to the trainer
      split (str) [optional, Default='train']: specify which
        dataset split to load. Useful for validating results
    '''
    with open(trainer_fp, 'rb') as f:
        trainer = pickle.load(f)
        f.close()
        
    if 'train' in split.lower():
        dataloader = trainer.data.train_dl
    elif 'val' in split.lower():
        dataloader = trainer.data.val_dl
    elif 'test' in split.lower():
        dataloader = trainer.data.test_dl
    else:
        print('Specify split to later allow for bootstrapping of data loader and testing of relations')
    dataloader.num_workers = 1
    for i, batch in enumerate(dataloader):
        X_mb = batch['x']
        y_mb = batch['y']
        idx_mb = batch['id']

        # grab md for mb
        for ii, unique_id in enumerate(idx_mb):
            if ii == 0:
                dt = pd.DataFrame(trainer.data.data['data'][unique_id]['md'], index=[unique_id])
            else:
                dt = dt.append(pd.DataFrame(trainer.data.data['data'][unique_id]['md'], index=[unique_id]))

        if i==0:
            X = X_mb[:, 0, :] # activity only
            Y = y_mb
            idx = idx_mb
            md = dt

        else:
            X = torch.cat((X, X_mb[:, 0, :]), dim=0)
            Y = torch.cat((Y, y_mb), dim=0)
            idx = idx + idx_mb
            md = md.append(dt)

    # clean up
    if md.index.to_list() == idx:
        md['GA'] = Y.numpy()
        del Y, idx
    else:
        warnings.warn('Unique identifier data sample mismatch')
    
    return X, md, trainer

        
def pairwise_DTW(X, verbose=False):
    if verbose:
        print('\nStarting pdist calculation')
        tic = time.time()
    def pdtw(mat):
        return pdist(mat, lambda u, v: fastdtw.fastdtw(u, v)[0])
    D = squareform(pdtw(X))
    if verbose:
        print('N={}\tT={}\t{:.0f}s'.format(X.shape[0], X.shape[1], time.time() - tic))
    return D

def embed(D, method='umap', metric='precomputed', verbose=False):
    '''
    Arguments:
      D (np.ndarray): shape (N x N) if distance matrix OR (N x M) for feature mat
      metric (str) [optional, Default='precomputed']: indicate whether D is distance or feat mat
    '''
    if verbose:
        print('\nStarting embedding ({} method)'.format(method))
        tic = time.time()
    if 'umap' in method.lower():
        data_embedding = umap.UMAP(metric=metric).fit_transform(D)
    else:
        raise NotImplementedError
    if verbose:
        print('N={}\t{:.0f}s'.format(D.shape[0], time.time() - tic))
    return data_embedding
        
def rawdata_umap_dtw(trainer_fp):
    X, md = data_from_trainer(trainer_fp)
    D = pairwise_DTW(X, verbose=True)
    data_umap = embed(D, verbose=True)
    return {'X': X, 'md': md, 'data_umap': data_umap, 'D': D}
    
def leiden_clustering(D, gamma=0.5):
    '''
    Arguments:
      D (np.ndarray): square distance matrix shape (N, N) with 0s in diag
      gamma (float): resolution parameter for clustering
      
    Returns:
      membership (list): cluster assignment
    '''

    A = NearestNeighbors(n_neighbors=5, metric='precomputed', n_jobs=16).fit(D).kneighbors_graph(D)
    G = ig.Graph.Adjacency(A.astype(int), mode='undirected')
    membership = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter=gamma).membership
    return membership
