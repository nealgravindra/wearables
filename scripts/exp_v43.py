import sys
sys.path.append('/home/ngrav/project/')
from wearables.scripts import utils as wearutils
from wearables.scripts import data_v42 as weardata
from wearables.scripts import model as wearmodels
from wearables.scripts import train_v42 as weartrain
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
    if 'cnn' in exp.lower():
        net = CNN()
    elif 'inceptiontime' in exp.lower() or 'it' in exp.lower():
        net = IT()
    elif 'lstm' in exp.lower():
        net = LSTM()
    elif 'gru' in exp.lower():
        net = GRU()
    if 'l1' in exp.lower():
        criterion = weartrain.MSEL1(lambda_l1=0.01)
    
    # trainer
    if 'rand' in exp.lower():
        trainer = weartrain.train(
            net, exp=exp,
            criterion=criterion, 
            trial=trial,
            batch_size=64,
            nb_epochs=400,
            lr=1e-6,
            lambda_l2=1e-3,
            patience=None,
            min_nb_epochs=0,
            shuffle_label=True,
            out_file='/home/ngrav/project/wearables/results/train_v43.csv',
            model_path='/home/ngrav/scratch/wearables_model_zoo',
            device=torch.device('cuda:{}'.format(cuda_nb)))
    else:
        trainer = weartrain.train(
            net, exp=exp,
            criterion=criterion, 
            trial=trial,
            batch_size=64,
            nb_epochs=10000,
            lr=1e-6,
            lambda_l2=1e-3,
            patience=500,
            min_nb_epochs=400,
            out_file='/home/ngrav/project/wearables/results/train_v43.csv',
            model_path='/home/ngrav/scratch/wearables_model_zoo',
            device=torch.device('cuda:{}'.format(cuda_nb)))
    trainer.fit()
    
    # eval
    evaluation = weareval.eval_trained(
        trainer, 
        out_file='/home/ngrav/project/wearables/results/eval_test_v43.csv')
    print('{} results:'.format(exp))
    print('--------')
    print(evaluation.eval_performance)
