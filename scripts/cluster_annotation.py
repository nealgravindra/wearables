'''cluster_annotation.py

Description: 
  The script prints out a dict of the metric obs/expected ratio ( (obs - exp)/exp ) 
    for particular values, the p-value for the stat hyp test in a one-vs.-rest cluster
    assignment scheme, and its evaluation on the test set (to see if the cluster annotation
    holds). 
'''

import numpy as np
import pandas as pd

import sys
sys.path.append('/home/ngrav/project/')
