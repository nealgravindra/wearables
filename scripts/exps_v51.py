import sys
sys.path.append('/home/ngrav/project/')
from wearables.scripts import utils as wearutils
from wearables.scripts import data_v42 as weardata
from wearables.scripts import model as wearmodels
from wearables.scripts import train as weartrain
from wearables.scripts import eval_v42 as weareval

import torch
import torch.nn as nn
import numpy as np

def CNN():
    net = wearmodels.CNN(2, 10080, 1, [(1, 32), (1, 64), (1, 128), (3, 256)])
    return net

def IT(nb_layers=9):
    net = wearmodels.InceptionTime(2, 1, 
                                   bottleneck=1, 
                                   kernel_size=[96, 32, 4],
                                   nb_filters=32, 
                                   residual=True, 
                                   nb_layers=nb_layers)
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
    
    parser.add_argument('--exp', type=str, default='dev', help='Experiment name')
    parser.add_argument('--trial', type=str, default=0, help='Relicate number')
    parser.add_argument('--cuda_nb', type=int, default=0, help='Cuda device nb')
    parser.add_argument('--model_type', type=str, default='InceptionTime', help='specify one of [InceptionTime, IT, GRU, CNN, LSTM]')
    parser.add_argument('--nb_layers', type=int, default=9, help='specify the number of layers for InceptionTime. Ignored if other model used')
    parser.add_argument('--criterion', type=str, default='MSE', help='specify one of MSE or MSEL1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=400, help='if 0, assume there is no patience added')
    parser.add_argument('--min_nb_epochs', type=int, default=400, help='if patience, do not start scheme until after this epoch passed')
    parser.add_argument('--scheduler', type=bool, default=False, help='use learning rate scheduler (if True, lr becomes initial)')
    parser.add_argument('--lambda_l1', type=float, default=1e-6, help='L1 regularization')
    parser.add_argument('--lambda_l2', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--aug_mode', type=str, default=['random'], help='specify one of None, ["random"], ["all"]')
    parser.add_argument('--aug_per_epoch', type=bool, default=False, help='for aug_mode, apply per epoch (TRUE) or per mb (FALSE)')
    parser.add_argument('--label_shuffle', type=bool, default=False, help='randomize labels to get null output for comparison')
    parser.add_argument('--model_path', type=str, default='/home/ngrav/scratch/wearables_model_zoo')
    parser.add_argument('--train_out_file', type=str, default='/home/ngrav/project/wearables/results/train_v51.csv')
    parser.add_argument('--eval_out_file', type=str, default='/home/ngrav/project/wearables/results/eval_v51.csv')
    args = parser.parse_args()
    if 'none' in args.model_path.lower():
        args.model_path = None
    if 'none' in args.model_path.lower():
        args.out_file = None
    args.exp = '{}_{}'.format(args.exp, args.model_type)
    
    # select model
    if 'cnn' in args.model_type.lower():
        net = CNN()
    elif 'inceptiontime' in args.model_type.lower() or 'it' in args.model_type.lower():
        net = IT()
    elif 'lstm' in args.model_type.lower():
        net = LSTM()
    elif 'gru' in args.model_type.lower():
        net = GRU()
    else:
        print('Inappropriate model type selected. Exiting')
        sys.exit() 
        
    # select loss fx and optimizer
    if args.criterion.lower() == 'mse':
        criterion = nn.MSELoss()
    elif args.criterion.lower() == 'msel1':
        criterion = weartrain.MSEL1(lambda_l1=args.lambda_l1)
    else:
        print('Loss fx not implemented. Exiting')
        sys.exit()
        
    # trainer
    trainer = weartrain.train(
        net, 
        exp=args.exp,
        trial=args.trial,
        batch_size=args.batch_size,
        nb_epochs=args.nb_epochs,
        lr=args.lr,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        criterion=criterion,
        scheduler=args.scheduler,
        aug_mode=args.aug_mode,
        aug_per_epoch=args.aug_per_epoch,
        patience=args.patience,
        min_nb_epochs=args.min_nb_epochs,
        shuffle_label=args.label_shuffle,
        out_file=args.train_out_file,
        model_path=args.model_path,
        device=torch.device('cuda:{}'.format(args.cuda_nb)))
    trainer.fit()
    
    # eval
    evaluation = weareval.eval_trained(
        trainer, 
        out_file=args.eval_out_file)
    
    # print out all parsed args to log file
    print('Passed arguments:')
    print(' ', args)
    
    print('\n{} results:'.format(args.exp))
    print('--------')
    print(evaluation.eval_performance)
