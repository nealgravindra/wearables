import torch
import numpy as np
import math
import torch.nn as nn

# simulate ts
class simulation():
    def __init__(self, N, dim=10, T=10800, a=math.pi, b=math.pi/2):
        self.dim = dim
        self.T = T
        self.a = a
        self.b = b
        self.N = N
    
    def sinusoidal_ts(self):
        X = torch.cat(self.dim*[torch.arange(self.T).reshape(-1, 1)/(2*math.pi)], 1)
        X = X.unsqueeze(0).repeat(self.N, 1, 1)
        self.X = torch.sin(self.a*X + self.b) + torch.empty(X.shape).normal_(mean=0, std=1)

def sim_sin_ts():
    sim = simulation(N=1000)
    sim.sinusoidal_ts()
    return sim.X # shape=(N, T, n_dim)

class EMA(nn.Module):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 -self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

if __name__ == '__main__':
    X = sim_sin_ts()

