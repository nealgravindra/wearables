'''
'preprocessing.py' extracts information, and transforms the data from downloaded metadata (.csv file) and raw 
  actigraphy data (stored in .csv files within a zipped file in ~/wearables/data/raw/) for use in a dictionary.

Meant to run once, to create the dictionary, or to serve as a record for preprocessing steps (for later enhancements).

Example:
  $ python -u preprocessing.py 
'''


import glob
import numpy as np
import os
import time

class timer(object):
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tic = time.time()

    def stop(self):
        self.times.append(time.time() - self.tic)
        return self.times[-1]

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def get_fnames(data_fp='/home/ngr/wearables/data/MOD_1000_Woman_Activity_Data'):
    glob.glob(data_fp)
    
if __name__ == '__main__':
    ticker = timer()
