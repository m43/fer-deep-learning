from collections import OrderedDict

import torch
import torch.nn as nn


class PTDeep(nn.Module):
    activation_name_to_fn = {
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        'tanh': torch.tanh,
    }

    def __init__(self, neruons_per_layer, activation_name):
        super().__init__()
        assert (len(neruons_per_layer) > 1)
        self.neruons_per_layer = neruons_per_layer
        self.activation_name = activation_name
        self.activation_fn = self.activation_name_to_fn[activation_name]

        self.weights = []
        self.biases = []
        for i in range(1, len(neruons_per_layer)):
            n_in, n_out = neruons_per_layer[i - 1], neruons_per_layer[i]
            self.weights.append(nn.Parameter(torch.randn((n_in, n_out))))
            self.biases.append(nn.Parameter(torch.zeros(n_out)))

        self.weights = nn.ParameterList(self.weights)
        self.biases = nn.ParameterList(self.biases)

    def forward(self, X):
        hidden = X
        for i in range(len(self.weights)):
            scores = hidden @ self.weights[i] + self.biases[i]
            if i == len(self.weights) - 1:
                continue
            hidden = self.activation_fn(scores)

        probs = torch.softmax(scores, dim=1)
        return probs

    def get_loss(self, X, Yoh_, weight_decay):
        Y = self.forward(X)
        nLL = -(Yoh_ * torch.log(Y)).sum(axis=1).mean()

        L2_penalty = 0
        for w in self.weights:
            L2_penalty += weight_decay * (w ** 2).sum()

        loss = nLL + L2_penalty
        return loss


class PTDeep2(nn.Module):
    activation_name_to_nn = {
        "relu": torch.nn.ReLU,
        "sigmoid": torch.nn.Sigmoid,
        'tanh': torch.nn.Tanh,
    }

    def __init__(self, neurons_per_layer, activation_name):
        super().__init__()
        assert (len(neurons_per_layer) > 1)
        self.neurons_per_layer = neurons_per_layer
        self.activation_name = activation_name

        layers = OrderedDict()
        for i in range(1, len(neurons_per_layer)):
            n_in, n_out = neurons_per_layer[i - 1], neurons_per_layer[i]
            layers[f"ll_{i}"] = nn.Linear(in_features=n_in, out_features=n_out, bias=True)

            if i == len(self.neurons_per_layer) - 1:
                layers[f"a_{i}"] = torch.nn.Softmax()
            else:
                layers[f"a_{i}"] = self.activation_name_to_nn[activation_name]()

        self.seq = nn.Sequential(layers)

    def forward(self, X):
        return self.seq(X)

    def get_loss(self, X, Yoh_, weight_decay):
        Y = self.forward(X)
        nLL = -(Yoh_ * torch.log(Y)).sum(axis=1).mean()

        # NB: nn.NLLLoss could have been used if Y_ was given instead of Yoh_
        #     Don't wanna 1. change the signature 2. decrease speed
        #     The following would also work:
        # Y_ = Yoh_.argmax(dim=-1)
        # nLL = nn.NLLLoss()(torch.log(Y), Y_)
        # print(nLL, nLL2)  # Same

        L2_penalty = weight_decay * sum([(x.weight ** 2).sum() for x in self.seq if type(x) == nn.Linear])

        return nLL + L2_penalty
