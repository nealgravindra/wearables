from distutils.core import setup
from Cython.Build import cythonize

import numpy as np


setup(name='dtw', 
      ext_modules=cythonize('/home/ngrav/project/wearables/scripts/dtw.pyx'), 
      include_dirs=['.', '/home/ngrav/project/wearables/scripts'], )