import pandas as pd
import pickle
import time
import importlib
import numpy as np
import importlib


import sys
sys.path.append('/home/ngr/gdrive/wearables/scripts')
import data as weardata
import utils as wearutils

import matplotlib.pyplot as plt
import seaborn as sns

def md_prediction(data, md, model='RFts', out_file=None):
    '''Predict variables in the metadata, given a voi dict.

    Arguments:
      model (str): implemented models are RFts (short for RF Time Series), kNN
    '''
    # load additional modules
    from sklearn.model_selection import KFold

    # load data
    timer = wearutils.timer()
    timer.start()
    data, md = weardata.load_data_md()
    print('Data and md loaded in {:.0f}-s'.format(timer.stop()))

    for i, label in enumerate(md.columns()):
        if label != 'record_id':
            X_train, y_train, X_test, y_test = get_Xy(data, md, label=label)

            # clf or reg
            if isinstance(y_train[0], float):
                continue
            else:
