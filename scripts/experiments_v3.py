import pandas as pd
import pickle
import time
import importlib
import numpy as np
import importlib
import datetime


import sys
sfp = '/home/ngr4/project/'
sys.path.append(sfp)
from wearables.scripts import data_v3 as weardata
from wearables.scripts import utils as wearutils
from wearables.scripts import train_v3 as weartrain


def InceptionTime_allmd(trial, colnum):
    # 377 targets
    with open('/home/ngr4/project/wearables/data/processed/targets_210929.pkl', 'rb') as f:
        target_list = pickle.load(f)
        f.close()
    target = target_list['colnum']
    trainer = weartrain.InceptionTime_trainer(
        exp='ITv3_'+target, trial=trial, new_split=False,
        out_file='/home/ngr4/project/wearables/results/InceptionTimev3_allmd.csv',
        target=target)
    trainer.fit()
    return trainer.eval_test()
    
def InceptionTime_GA(trial):
    trainer = weartrain.InceptionTime_trainer(
        exp='ITv3_GA', trial=trial, 
        model_path='/home/ngr4/scratch60',
        new_split=True,
        out_file='/home/ngr4/project/wearables/results/GA_robustness_tests.csv',
        target='GA')
    trainer.fit()
    return trainer.eval_test()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Experiment name')
    parser.add_argument('--trial', type=str, help='Relicate number')
    parser.add_argument('--colnum', type=int, help='Column number to specify target from md')

    args = parser.parse_args()
    if args.exp is None:
        exp = 'ITv3_GA' 
    else:
        exp = args.exp
    if args.trial is None:
        trial = 0 # assume dev
    else:
        trial = args.trial
    if args.colnum is not None:
        colnum = args.colnum

    if exp == 'ITv3_GA':
        res = InceptionTime_GA(trial)
    elif exp == 'allmd':
        res = InceptionTime_allmd(trial, colnum)
    else:
        print('Exp {} not implemented.'.format(exp))
