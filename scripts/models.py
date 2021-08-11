import torch
from torch import nn
import torch.nn.functional as F

####
# REF:
## 1. official implementation of InceptionTime (in tf):
##      https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
## 2. additional reference for PyTorch implementation: https://mohcinemadkour.github.io/posts/2019/10/Machine%20Learning,%20timeseriesAI,%20Time%20Series%20Classification,%20fastai_timeseries,%20TSC%20bechmark/
##
####



class Inception(nn.Module):
    def __init__(self, in_channels, bottleneck=16, kernel_size=60, nb_filters=32):
        '''Inception
        '''
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, 1, padding='same') if bottleneck and in_channels > 0 else lambda x: x# (bug) this is not really a bottleneck layer, REMOVE following: if bottleneck and in_channels > 1 else lambda x: x
        conv_layers = []
        kss = [kernel_size // (2**i) for i in range(3)] # (enhancement) 360 down to 5 min, for a total of 9 blocks?
        d_mts = bottleneck or in_channels
        for i in range(len(kss)):
            conv_layers.append(
                nn.Conv1d(d_mts, nb_filters, kernel_size=kss[i], padding='same')
                )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1) # (enhancement) to kernel_size=5?
        self.conv = nn.Conv1d(in_channels, nb_filters, kernel_size=1)
        self.BN = nn.BatchNorm1d(nb_filters * 4)
        self.activation = nn.ReLU() # (bug) original act is ReLU, change to LeakyReLU

    def forward(self, x): # x.shape=(N, Cin, L) where N=batch size, Cin=inpuit dim, L=num. timepoints
        input_tensor = x
        x = self.bottleneck(input_tensor)
        for i in range(3):
            out_ = self.conv_layers[i](x)
            if i == 0:
                out = out_
            else:
                out = torch.cat((out, out_), 1)
        mp = self.conv(self.maxpool(input_tensor))
        inc_out = torch.cat((out, mp), 1)
        return self.activation(self.BN(inc_out)) # order of BN supported by https://arxiv.org/abs/1502.03167 section 3.2

def shortcut_layer(in_channels, out_channels): # not a class in official implementation (https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py)
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1),
                         nn.BatchNorm1d(out_channels))

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, bottleneck=16,
                 kernel_size=60, nb_filters=32, residual=True, nb_layers=6):
        super().__init__()
        self.residual = residual
        self.nb_layers = nb_layers
        inception_layers = []
        residual_layers = []
        l_res = 0
        for l_inc in range(nb_layers):
            inception_layers.append(
                Inception(in_channels if l_inc==0 else nb_filters * 4,
                          bottleneck=bottleneck if l_inc > 0 else 0,
                          kernel_size=kernel_size,
                          nb_filters=nb_filters)
                          )
            if self.residual and l_inc % 3 == 2:
                residual_layers.append(shortcut_layer(in_channels if l_res == 0 else nb_filters * 4, nb_filters * 4))
                l_res += 1
            else:
                res_layer = residual_layers.append(None)
        self.inception_layers = nn.ModuleList(inception_layers)
        self.residual_layers = nn.ModuleList(residual_layers)
        self.activation = nn.ReLU()

    def forward(self, x):
        res = x
        for l in range(self.nb_layers):
            x = self.inception_layers[l](x)
            if self.residual and l % 3 == 2:
                res = self.residual_layers[l](res)
                x += res
                res = x
                x = self.activation(x) # (question) BN here?
        return x

class InceptionTime(nn.Module):
    '''Modified for regression.

    '''
    def __init__(self, in_channels, out_channels, bottleneck=8, kernel_size=60,
                 nb_filters=16, residual=True, nb_layers=6):
        super().__init__()
        self.block = InceptionBlock(in_channels, bottleneck=bottleneck,
                                    kernel_size=kernel_size, nb_filters=nb_filters,
                                    residual=residual, nb_layers=nb_layers)
        self.gap = nn.AdaptiveAvgPool1d(1) # gap: mts to 1 number
        self.fc = nn.Linear(nb_filters * 4, out_channels) # channels to output_dim

    def forward(self, x):
        x = self.block(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x

def chk_InceptionTime():
    # data
    N = 32 # batch size
    Cin = 1
    L = 10801
    xt = torch.rand((N, Cin, L))

    # architecture
    output = InceptionTime(1, 1)(xt)
    print('Data shape: ({}, {})'.format(*xt.shape))
    print('output shape: ({}, {})'.format(*output.shape))

if __name__ == '__main__':
    chk_InceptionTime()
