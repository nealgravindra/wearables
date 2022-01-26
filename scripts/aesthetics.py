import seaborn as sns
import matplotlib.pyplot as plt


models = [
    'kNN',
    'RandomForest',
    'Gradient Boosting',
    'TimeSeriesForest',
    'GRU',
    'VGG-1D',
    'InceptionTime (Random)',
    'Ours',
]
model_cmap = {k:plt.get_cmap('cubehelix', 9)(i) for i, k in enumerate(models)}

slp_binary_color = '#4297A0'

ptb_cmap = {False: '#2F5061', True: '#E57F84'}

split_cmap = {'train': '#4297A0', 'test': '#F4EAE6'}

def p_encoder(p):
    if p > 0.05:
        label = '' # n.s.
    elif p <= 0.001:
        label = '***'
    elif p <= 0.05 and p > 0.01:
        label = '*'
    elif p <= 0.01 and p > 0.001:
        label = '**'
    else: 
        label = 'Unclassified'
    return label

traintest_ptbyn_cmap = {'train_ptby': '#F4EAE6',
                        'train_ptbn': '#4297A0',
                        'test_ptby': '#E57F84',
                        'test_ptbn': '#2F5061'}
