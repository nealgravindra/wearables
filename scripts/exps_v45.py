import sys
sys.path.append('/home/ngrav/project/')
from wearables.scripts import utils as wearutils
from wearables.scripts import data_v42 as weardata
from wearables.scripts import model as wearmodels
from wearables.scripts import train as weartrain
from wearables.scripts import eval_v42 as weareval

import torch
import numpy as np

def CNN():
    net = wearmodels.CNN(2, 10080, 1, [(1, 32), (1, 64), (1, 128), (3, 256)])
    return net

def IT():
    net = wearmodels.InceptionTime(2, 1, 
                                   bottleneck=1, 
                                   kernel_size=[96, 32, 4],
                                   nb_filters=32, 
                                   residual=True, 
                                   nb_layers=9)
    return net

def LSTM():
#     net = wearmodels.LSTM(2, 64, 3, 10080, 1)
    return NotImplementedError # need to add addl_out args

def GRU():
    net = wearmodels.GRU(2, 64, 3, 10080, 1)
    return net

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Experiment name')
    parser.add_argument('--trial', type=str, help='Relicate number')
    parser.add_argument('--cuda_nb', type=int, help='Cuda device nb')

    args = parser.parse_args()
    exp = args.exp
    trial = args.trial
    cuda_nb = 0 if args.cuda_nb is None else args.cuda_nb
    
    # select model
    batch_size = 64
    if 'cnn' in exp.lower():
        net = CNN()
    elif 'inceptiontime' in exp.lower() or 'it' in exp.lower():
        net = IT()
        batch_size = 32
    elif 'lstm' in exp.lower():
        net = LSTM()
    elif 'gru' in exp.lower():
        net = GRU()
    if 'randaug' in exp.lower():
        aug_mode = ['random']
    elif 'allaug' in exp.lower():
        aug_mode = ['all']
    else: 
        print('Warning. No augmentation mode selected')
    if 'perepoch' in exp.lower():
        aug_per_epoch = True
    else:
        aug_per_epoch = False
        
    
    # trainer
    if 'labelshuffle' in exp.lower():
        trainer = weartrain.train(
            net, exp=exp,
            trial=trial,
            batch_size=batch_size,
            nb_epochs=400,
            lr=1e-5,
            lambda_l2=0.001,
            aug_mode=aug_mode,
            aug_per_epoch=aug_per_epoch,
            patience=None,
            min_nb_epochs=0,
            shuffle_label=True,
            out_file='/home/ngrav/project/wearables/results/train_v45.csv',
            model_path='/home/ngrav/scratch/wearables_model_zoo',
            device=torch.device('cuda:{}'.format(cuda_nb)))
    else:
        trainer = weartrain.train(
            net, exp=exp,
            trial=trial,
            batch_size=batch_size,
            nb_epochs=10000,
            lr=1e-5,
            lambda_l1=(5e-4)*0.5,
            lambda_l2=5e-4,
            aug_mode=aug_mode,
            aug_per_epoch=aug_per_epoch,
            patience=400,
            min_nb_epochs=400,
            out_file='/home/ngrav/project/wearables/results/train_v45.csv',
            model_path='/home/ngrav/scratch/wearables_model_zoo',
            device=torch.device('cuda:{}'.format(cuda_nb)))
    trainer.fit()
    
    # eval
    evaluation = weareval.eval_trained(
        trainer, 
        out_file='/home/ngrav/project/wearables/results/eval_test_v45.csv')
    print('{} results:'.format(exp))
    print('--------')
    print(evaluation.eval_performance)
