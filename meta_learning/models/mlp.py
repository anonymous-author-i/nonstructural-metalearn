import torch
import torch.nn as nn


class MLP_withoutFC(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout, **kwargs):
        """
        MLP without full-connected layer
        """
        super(MLP_withoutFC, self).__init__()
        self.model = self.make(input_size, hidden_sizes, dropout)

    def make(self, input_size, hidden_sizes, dropout):
        layers = []
        if hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            for i in range(len(hidden_sizes) - 1):
                # exclude the input layer
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, x, args=None):
        x = self.model(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout, **kwargs):
        super(MLP, self).__init__()
        self.model = self.make(input_size, hidden_sizes, output_size, dropout)

    def make(self, input_size, hidden_sizes, output_size, dropout):
        layers = []
        if hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            for i in range(len(hidden_sizes) - 1):
                # exclude the input layer
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_sizes[-1], output_size))
        else:
            layers.append(nn.Linear(input_size, output_size))

        return nn.Sequential(*layers)

    def forward(self, x, args=None):
        x = self.model(x)
        return x