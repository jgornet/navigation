import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from math import sqrt


class RNN(nn.Module):
    def __init__(self, N=100, N_in=2, N_out=2, steps=20):
        super().__init__()
        
        self.steps = steps

        self.w_x = nn.Linear(N_in, N)
        nn.init.normal_(self.w_x.weight, mean=0, std=1/sqrt(N_in))
        nn.init.constant_(self.w_x.bias, 0)

        self.w_v = nn.Linear(N_in, N, bias=False)
        nn.init.normal_(self.w_v.weight, mean=0, std=1/sqrt(N_in))

        self.w_out = nn.Linear(N, N_out, bias=True)
        nn.init.normal_(self.w_out.weight, mean=0, std=1/sqrt(N))
        nn.init.constant_(self.w_out.bias, 0)

    def loss(self, label, inputs):
        return F.mse_loss(label, inputs)

    def step(self, x_i, v):
        dxdt = self.w_out(torch.relu(self.w_x(x_i) + self.w_v(v)))
        x_f = x_i + dxdt

        return x_f
