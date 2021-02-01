import torch
import torch.nn as nn
import numpy as np

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """
        Arguments:
        - D: dimensions of each datapoint
        - C: number of classes
        """
        super().__init__()
        self.w = nn.Parameter(torch.tensor(np.random.randn(D,C), dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(C, dtype=torch.float64))

    def forward(self, X):
        scores = X @ self.w + self.b
        probs = torch.softmax(scores, dim=1)
        return probs

    def get_loss(self, X, Yoh_):
        Y = self.forward(X)
        nLL = -(Yoh_*torch.log(Y)).sum(axis = 1).mean()
        return nLL
