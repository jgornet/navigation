import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from math import sqrt


class GRU(nn.Module):
    def __init__(self, hidden_size=100):
        super().__init__()

        self.gru = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.w_out = nn.Linear(hidden_size, 2, bias=True)

    def initialize_h(self, x):
        bias = self.w_out.bias
        W = self.w_out.weight
        b = torch.linalg.solve(W @ W.t(), (x - bias).t())
        h = b.t() @ W

        return h.unsqueeze(0)

    def forward(self, x_i, v):
        h_0 = self.initialize_h(x_i)
        h, h_n = self.gru(v, h_0)
        x = self.w_out(h)

        return x


class RNN(nn.Module):
    def __init__(self, hidden_size=100):
        super().__init__()
        self.rnn = nn.RNN(input_size=2, hidden_size=hidden_size,
                          batch_first=True, nonlinearity='relu')
        self.w_out = nn.Linear(hidden_size, 2, bias=True)

    def initialize_h(self, x):
        bias = self.w_out.bias
        W = self.w_out.weight
        b = torch.linalg.solve(W @ W.t(), (x - bias).t())
        h = b.t() @ W

        return h.unsqueeze(0)

    def forward(self, x_i, v):
        h_0 = self.initialize_h(x_i)
        h, h_n = self.rnn(v, h_0)
        x = self.w_out(h)

        return x
