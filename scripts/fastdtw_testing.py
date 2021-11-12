# REF: https://github.com/slaypni/fastdtw

import fastdtw
import torch
from sklearn.neighbors import KNeighborsRegressor
import time

def generate_data(N, L, D):
    '''
    Arguments:
      N (int): number of samples
      L (int): sequence length
      D (int): dimensionality

    '''
    return torch.rand(N, L, D)

def DTW(x, y):
    return fastdtw.fastdtw(x, y)[0]

knndtw = KNeighborsRegressor(n_neighbors=5, algorithm='ball_tree',
                             metric=DTW, weights='distance')

X = generate_data(32, 10080, 1).squeeze()


knndtw.fit(X, torch.rand(X.shape[0],))

# speed test 

results = {'n': [], 'sec': [], 'L': []}
for n in [32, 256, 512, 1024, 2048]:
    X = generate_data(n, 10080, 1).squeeze()
    tic = time.time()
    knndtw = KNeighborsRegressor(n_neighbors=5, algorithm='ball_tree',
                             metric=DTW, weights='distance')
    knndtw.fit(X, torch.rand(X.shape[0],))
    preds = knndtw.predict(X)
    results['sec'].append(time.time() - tic)
    results['n'].append(n)
    results['L'].append(X.shape[1])