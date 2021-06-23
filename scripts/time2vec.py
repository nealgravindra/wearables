import torch
from torch import nn

# periodic activations
def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], 1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
    
    
class time2vec(nn.Module):
    def __init__(self, hidden_dim, output_dim, activation='sin'):
        super().__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, hidden_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    sineact = SineActivation(1, 64)
    cosact = CosineActivation(1, 64)

    print(sineact(torch.Tensor([[7]])).shape)
    print(cosact(torch.Tensor([[7]])).shape)