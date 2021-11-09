import pickle
import numpy as np
import pandas as pd
import os
import glob
import warnings
import time

import sys
sfp = '/home/ngrav/project/'
sys.path.append(sfp)
from wearables.scripts import utils as wearutils


import torch

class raw2np():
    '''

    The data was sent as .csv files, annotated with the GA and pid
      in the filepath. The csv file contains activity counts and indicators
      of light, as well as the datetime. We want to specify the number of weeks
      that this is available for and pair it with available metadata. Then, we pare 
      down the metadata file by transforming it in specific ways, using manual dictionary
      entries. 
    '''
    def __init__(self, 
        md_filter_dict=None, 
        nb_weeks=2,  
        data_folder_path='/home/ngrav/project/wearables/data/raw/MOD 1000 Woman Activity Data-20210707T213505Z-001/MOD 1000 Woman Activity Data/'):
        self.data_folder_path = data_folder_path
        print('i am here. hello, world')
        
