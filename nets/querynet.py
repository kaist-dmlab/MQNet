import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryNet(nn.Module):
    def __init__(self, input_size=2, inter_dim=64):
        super().__init__()

        self.linear_1 = nn.Linear(input_size, inter_dim)
        self.linear_2 = nn.Linear(inter_dim, 1)

    def forward(self, X):
        out = self.linear_1(X)
        out = self.linear_2(out)
        return out