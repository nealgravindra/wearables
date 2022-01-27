import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline
# plot settings
plt.rc('font', size = 9)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['legend.frameon']=False
plt.rcParams['axes.grid']=False
plt.rcParams['legend.markerscale']=1
plt.rcParams['savefig.dpi'] = 600
sns.set_style("ticks")


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


errgrp_cmap = {'lt10wks': '#4297A0', # green
               'Higher-than-actual': '#E57F84', #red
               'Lower-than-actual': '#2F5061', # blue
               }