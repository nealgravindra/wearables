import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('/home/ngrav/project/')
from wearables.scripts import utils as wearutils
from wearables.scripts import data_v42 as weardata
from wearables.scripts import models_v42 as wearmodels
from wearables.scripts import train_v42 as weartrain
from wearables.scripts import eval_v42 as weareval

import torch

import numpy as np

def CNN():
    net = wearmodels.CNN(2, 10080, 1, [(1, 32), (1, 64), (1, 128), (3, 256)])
    return net

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Experiment name')
    parser.add_argument('--trial', type=str, help='Relicate number')

    args = parser.parse_args()
    exp = args.exp
    trial = args.trial
    
    # select model
    if 'cnn' in exp.lower():
        net = CNN()
    elif 'inceptiontime' in exp.lower():
        raise NotImplementedError
    if 'l1' in exp.lower():
        criterion = weartrain.MSEL1()
    else:
        criterion=nn.MSELoss()
    
    # train
    trainer = weartrain.train(
        net, exp=exp,
        criterion=criterion, 
        trial=trial,
        load_splits=None,
        n_epochs=5000,
        batch_size=32,
        lambda_l2=0.001,
        lr=1e-6,
        patience=500,
        device=torch.device('cuda:0'))
    trainer.fit()
    
    # eval
    evaluation = weareval.eval_trained(
        trainer, 
        out_file='/home/ngrav/project/wearables/results/eval_test_v42.csv')
    print('{} results:'.format(exp))
    print('--------')
    print(evaluation.results)
