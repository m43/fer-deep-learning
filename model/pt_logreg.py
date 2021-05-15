import numpy as np
import torch
import torch.nn as nn


class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """
        Arguments:
        - D: dimensions of each datapoint
        - C: number of classes
        """
        super().__init__()
        self.w = nn.Parameter(torch.tensor(np.random.randn(D, C), dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(C, dtype=torch.float64))

    def forward(self, X):
        scores = X @ self.w + self.b
        probs = torch.softmax(scores, dim=1)
        return probs

    def get_loss(self, X, Yoh_, weight_decay):
        Y = self.forward(X)
        nLL = -(Yoh_ * torch.log(Y)).sum(axis=1).mean()
        L2_penalty = weight_decay * (self.w ** 2).sum()
        loss = nLL + L2_penalty
        return loss
