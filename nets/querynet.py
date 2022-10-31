import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryNet(nn.Module):
    def __init__(self, input_size=2, inter_dim=64):
        super().__init__()

        W1 = torch.rand(input_size, inter_dim, requires_grad=True) #ones
        W2 = torch.rand(inter_dim, 1, requires_grad=True) #ones
        b1 = torch.rand(inter_dim, requires_grad=True) #zeros
        b2 = torch.rand(1, requires_grad=True) #zeros

        self.W1 = torch.nn.parameter.Parameter(W1, requires_grad=True)
        self.W2 = torch.nn.parameter.Parameter(W2, requires_grad=True)
        self.b1 = torch.nn.parameter.Parameter(b1, requires_grad=True)
        self.b2 = torch.nn.parameter.Parameter(b2, requires_grad=True)

        #print(self.W2) # all 1

    def forward(self, X):
        out = torch.sigmoid(torch.matmul(X, torch.relu(self.W1)) + self.b1)
        out = torch.matmul(out, torch.relu(self.W2)) + self.b2
        return out