import torch
from torch import nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_channels, bottleneck=16, kernel_size=60, nb_filters=32):
        '''Inception
        '''
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, 1, padding='same') if bottleneck and in_channels > 0 else lambda x: x# (bug) this is not really a bottleneck layer, REMOVE following: if bottleneck and in_channels > 1 else lambda x: x
        conv_layers = []
        if isinstance(kernel_size, list):
            kss = kernel_size 
        else:
            kss = [kernel_size // (2**i) for i in range(3)] if not isinstance(kernel_size, list) else kernel_size
        d_mts = bottleneck or in_channels
        for k in kss:
            conv_layers.append(
                nn.Conv1d(d_mts, nb_filters, kernel_size=k, padding='same')
                )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.maxpool = nn.MaxPool1d(5, stride=1, padding=2)
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
        return self.activation(self.BN(inc_out)) 

def shortcut_layer(in_channels, out_channels): # not a class in official implementation (https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py)
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1),
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
                 nb_filters=16, residual=True, nb_layers=8, regression=True):
        super().__init__()
        self.regression = regression
        self.block = InceptionBlock(in_channels, bottleneck=bottleneck,
                                    kernel_size=kernel_size, nb_filters=nb_filters,
                                    residual=residual, nb_layers=nb_layers)
        self.gap = nn.AdaptiveAvgPool1d(1) # gap: mts to 1 number
        self.fc = nn.Linear(nb_filters * 4, out_channels) # channels to output_dim

    def forward(self, x):
        x = self.block(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        if not self.regression: # then classification
            x = F.log_softmax(x, dim=1)
        return x
    
class LSTM(nn.Module):
    '''Basic comparison for a 2 layer, final hidden unit clf using LSTM
    '''
    def __init__(self, d_in, d_hidden, nb_layers, T, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.nb_layers = nb_layers
        self.T = T
        self.d_out = d_out
        
        # blocks
        self.lstm = nn.LSTM(d_in, d_hidden, 
                            num_layers=nb_layers, 
                            bidirectional=True, 
                            batch_first=True, 
                            dropout=0.5)
        self.pred = nn.Sequential(
            nn.Linear(d_in*d_hidden, d_in*d_hidden // 2),
            nn.LeakyReLU(),
            nn.Linear(d_in*d_hidden // 2, d_out)
        )
    
    def forward(self, X):
        # X shape (N, d_in, T) where d_in=data dim (activity counts and light intensity)
        _, (h_n, _) = self.lstm(X.transpose(2, 1))
        # https://discuss.pytorch.org/t/how-to-concatenate-the-hidden-states-of-a-bi-lstm-with-multiple-layers/39798/5
        ##  h_n = h_n.view(num_layers, num_directions, batch, hidden_size)
        h_n = h_n.reshape(self.nb_layers, 2, -1, self.d_hidden)[-1]
        h_n = h_n.reshape(-1, 2*self.d_hidden)
        return self.pred(h_n)
    
class GRU(nn.Module):
    '''Basic comparison for a 2 layer, final hidden unit clf using LSTM
    '''
    def __init__(self, d_in, d_hidden, nb_layers, T, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.nb_layers = nb_layers
        self.T = T
        self.d_out = d_out
        
        # blocks
        self.gru = nn.GRU(d_in, d_hidden, 
                            num_layers=nb_layers, 
                            bidirectional=True, 
                            batch_first=True, 
                            dropout=0.5)
        self.pred = nn.Sequential(
            nn.Linear(d_in*d_hidden, d_in*d_hidden // 2),
            nn.LeakyReLU(),
            nn.Linear(d_in*d_hidden // 2, d_out)
        )
    
    def forward(self, X):
        # X shape (N, d_in, T) where d_in=data dim (activity counts and light intensity)
        _, h_n = self.gru(X.transpose(2, 1))
        # https://discuss.pytorch.org/t/how-to-concatenate-the-hidden-states-of-a-bi-lstm-with-multiple-layers/39798/5
        ##  h_n = h_n.view(num_layers, num_directions, batch, hidden_size)
        h_n = h_n.reshape(self.nb_layers, 2, -1, self.d_hidden)[-1]
        h_n = h_n.reshape(-1, 2*self.d_hidden)
        return self.pred(h_n)
    
class CNN(nn.Module):
    '''Call it VGG-1D'''
    def __init__(self, in_channels, L, d_out, conv_arch):
        '''
        Arguments:
          L (int): length of input sequence, needed to determine how to flatten VGG output
          conv_arch (list of tuples): e.g., [(1, 64), (1, 128)] where the first value is 
            the nb_convs and the second value is the d_out
        '''
        super().__init__()
        self.d_out = d_out
        conv_blocks = []
        nb_halves = 0
        for (num_convs, out_channels) in conv_arch:
            nb_halves = nb_halves + num_convs
            conv_blocks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.vgg = nn.Sequential(*conv_blocks, nn.Flatten())
        self.pred = nn.Sequential(
            nn.Linear(out_channels*(L // (2**nb_halves)), 512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(512, d_out),
            )
        
    def vgg_block(self, nb_convs, in_channels, out_channels):
        layers = []
        for _ in range(nb_convs):
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1) 
                )
            layers.append(nn.LeakyReLU()) # add dropout?
            in_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
        
        
    def forward(self, X):
        # X.shape = (N, C, T)
        X = self.vgg(X)
        return self.pred(X)
        
    
class DeepSleepNet(nn.Module):
    def __init__(self, d_in, L, d_out, regression=True):
        '''
        Arguments:
          L (int): length of input sequence
        '''
        super().__init__()
        
        # represention 
        self.lofi = nn.Sequential(
            self.conv_blk(d_in, 64, 60, 5, 0), 
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(),
            self.conv_blk(64, 64, 60, 5, 0), 
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(),
            self.conv_blk(64, 128, 2, 1, 1),
            self.conv_blk(128, 128, 2, 1, 1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.hifi = nn.Sequential(
            self.conv_blk(d_in, 64, 5, 1, 0), 
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(),
            self.conv_blk(64, 64, 5, 1, 0), 
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(),
            self.conv_blk(64, 64, 5, 1, 0), 
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(),
            self.conv_blk(64, 64, 5, 1, 0), 
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(),
            self.conv_blk(64, 64, 5, 1, 0), 
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(),
            self.conv_blk(64, 128, 2, 1, 1),
            self.conv_blk(128, 128, 2, 1, 1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # sequential learning
        self.lofi_rnn = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
#         self.lofi_fc1(128, 256) # how would you do this? ave or for loop over the seq?
        self.hifi_rnn = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
#         self.hifi_fc1 = nn.Linear(128, 256)

        # pred
        self.pred = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, d_out)
        )
            
    def conv_blk(self, in_channels, nb_filters, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, nb_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(nb_filters), 
            nn.LeakyReLU())
        
    def forward(self, X):
        lo = self.lofi(X)
        hi = self.hifi(X)
        _, (lo, _) = self.lofi_rnn(lo.transpose(2, 1))
        lo = lo.view(2, 2, -1, 128)[-1, :, :, :].reshape(-1, 256) # (nb_layers, directions, N, L)
        _, (hi, _) = self.hifi_rnn(hi.transpose(2, 1))
        hi = hi.view(2, 2, -1, 128)[-1, :, :, :].reshape(-1, 256) # (nb_layers, directions, N, L)
        return self.pred(torch.cat((lo, hi), dim=-1))
    

class ConvLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        return NotImplementedError
        
        

def chk_InceptionTime():
    # data
    N = 32 # batch size
    Cin = 1
    L = 10801
    xt = torch.rand((N, Cin, L))

    # architecture
    output = InceptionTime(1, 3, regression=False)(xt)
    print('Data shape: ', xt.shape)
    print('output shape: ', output.shape)
    return output

if __name__ == '__main__':
    chk_InceptionTime()
