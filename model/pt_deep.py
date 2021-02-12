import math
import sys
from collections import OrderedDict

import torch
import torch.nn as nn


class PTDeep(nn.Module):
    activation_name_to_fn = {
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        'tanh': torch.tanh,
    }

    def __init__(self, neurons_per_layer, activation_name, init="normal", bn=False, bn_init_data=None, bn_epsilon=1e-5,
                 bn_momentum=0.9):
        super().__init__()
        assert (len(neurons_per_layer) > 1)
        assert (init in ["normal", "kaiming_uniform"])
        assert (not bn or bn_init_data is not None)

        self.neurons_per_layer = neurons_per_layer
        self.activation_name = activation_name
        self.activation_fn = self.activation_name_to_fn[activation_name]
        self.init = init
        self.bn = bn
        self.bn_epsilon = bn_epsilon

        self.weights = []
        self.biases = []
        bn_gammas = []
        bn_betas = []
        bn_mean_momenta, bn_var_momenta = [], []
        for i in range(1, len(neurons_per_layer)):
            n_in, n_out = neurons_per_layer[i - 1], neurons_per_layer[i]
            if init == "normal":
                self.weights.append(nn.Parameter(torch.randn((n_in, n_out))))
                self.biases.append(nn.Parameter(torch.zeros(n_out)))
            elif init == "kaiming_uniform":
                w = torch.empty((n_in, n_out))
                b = torch.empty(n_out)

                # as in nn.Linear.reset_parameters
                nn.init.kaiming_uniform_(w.T, a=math.sqrt(5))
                bound = 1 / math.sqrt(n_in)
                nn.init.uniform_(b, -bound, bound)

                self.weights.append(nn.Parameter(w))
                self.biases.append(nn.Parameter(b))
            if bn and i != len(neurons_per_layer) - 1:
                bn_gammas.append(nn.Parameter(torch.normal(1.0, 0.01, size=(n_out,))))
                bn_betas.append(nn.Parameter(torch.normal(0.0, 0.01, size=(n_out,))))
                bn_mean_momenta.append(nn.Parameter(torch.empty(n_out), requires_grad=False))
                bn_var_momenta.append(nn.Parameter(torch.empty(n_out), requires_grad=False))

        self.weights = nn.ParameterList(self.weights)
        self.biases = nn.ParameterList(self.biases)
        if bn:
            self.bn_gammas = nn.ParameterList(bn_gammas)
            self.bn_betas = nn.ParameterList(bn_betas)
            self.bn_mean_momenta = nn.ParameterList(bn_mean_momenta)
            self.bn_var_momenta = nn.ParameterList(bn_var_momenta)
            self.bn_momentum = 0.0
            # self.bn_momentum = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.forward(bn_init_data)
            self.bn_momentum = bn_momentum
            # self.bn_momentum += torch.tensor(bn_momentum)

    def forward(self, X):
        hidden = X
        for i in range(len(self.weights)):
            scores = hidden @ self.weights[i] + self.biases[i]

            if i == len(self.weights) - 1:  # TODO does this if statement slow down GPU utilisation?
                continue

            if self.bn:
                if self.training:
                    mean = scores.mean(dim=0)
                    var = torch.mean((scores - mean) ** 2, dim=0)
                    self.bn_mean_momenta[i] *= self.bn_momentum
                    self.bn_mean_momenta[i] += (1.0 - self.bn_momentum) * mean.clone().detach()
                    self.bn_var_momenta[i] *= self.bn_momentum
                    self.bn_var_momenta[i] += (1.0 - self.bn_momentum) * var.clone().detach()

                    scores_hat = (scores - mean) / (var + self.bn_epsilon)
                else:
                    scores_hat = (scores - self.bn_mean_momenta[i]) / (self.bn_var_momenta[i] + self.bn_epsilon)

                scores = self.bn_gammas[i] * scores_hat + self.bn_betas[i]

            hidden = self.activation_fn(scores)

        probs = torch.softmax(scores, dim=1)
        return probs

    def get_loss(self, X, Yoh_, weight_decay):
        Y = self.forward(X) + sys.float_info.epsilon

        nLL = -(Yoh_ * torch.log(Y)).sum(axis=1).mean()
        L2_penalty = weight_decay * sum([(w ** 2).sum() for w in self.weights])

        return nLL + L2_penalty


class PTDeep2(nn.Module):
    activation_name_to_nn = {
        "relu": torch.nn.ReLU,
        "sigmoid": torch.nn.Sigmoid,
        'tanh': torch.nn.Tanh,
    }

    def __init__(self, neurons_per_layer, activation_name, init="kaiming_uniform"):
        super().__init__()
        assert (len(neurons_per_layer) > 1)
        assert (init in ["normal", "kaiming_uniform"])

        self.neurons_per_layer = neurons_per_layer
        self.activation_name = activation_name
        self.init = init

        if init == "normal":
            weights = [None]
            biases = [None]
            for i in range(1, len(neurons_per_layer)):
                n_in, n_out = neurons_per_layer[i - 1], neurons_per_layer[i]
                weights.append(torch.transpose(torch.randn((n_in, n_out)), 0, 1))
                biases.append(torch.zeros(n_out))

        layers = OrderedDict()
        for i in range(1, len(neurons_per_layer)):
            n_in, n_out = neurons_per_layer[i - 1], neurons_per_layer[i]
            layers[f"ll_{i}"] = nn.Linear(in_features=n_in, out_features=n_out, bias=True)

            if init == "normal":
                layers[f"ll_{i}"].weight.data = weights[i]
                layers[f"ll_{i}"].bias.data = biases[i]

            if i == len(self.neurons_per_layer) - 1:
                layers[f"a_{i}"] = nn.Softmax()
            else:
                layers[f"a_{i}"] = self.activation_name_to_nn[activation_name]()

        self.seq = nn.Sequential(layers)

    def forward(self, X):
        return self.seq(X)

    def get_loss(self, X, Yoh_, weight_decay):
        if self.init == "normal":
            Y = self.forward(X) + sys.float_info.epsilon
        else:
            Y = self.forward(X)

        # NB: nn.NLLLoss could have been used if Y_ was given instead of Yoh_
        #     Don't wanna 1. change the signature 2. decrease speed
        #     The following would also work:
        # Y_ = Yoh_.argmax(dim=-1)
        # nLL = nn.NLLLoss()(torch.log(Y), Y_)
        # print(nLL, nLL2)  # Same

        nLL = -(Yoh_ * torch.log(Y)).sum(axis=1).mean()
        L2_penalty = weight_decay * sum([(x.weight ** 2).sum() for x in self.seq if type(x) == nn.Linear])

        return nLL + L2_penalty
