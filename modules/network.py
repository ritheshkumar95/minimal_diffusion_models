import torch
import torch.nn as nn


class FiLM1d(nn.Module):
    def __init__(self, hidden_dim, n_classes):
        super().__init__()
        self.emb = nn.Embedding(n_classes, hidden_dim * 2)
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False)

    def forward(self, x, y):
        out = self.norm(x)
        alpha, beta = self.emb(y).chunk(2, dim=-1)
        return alpha + out * (1 + beta)


class ToyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim),
                FiLM1d(hidden_dim, n_classes),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                FiLM1d(hidden_dim, n_classes),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            ]
        )

    def forward(self, x, y):
        for layer in self.model:
            if isinstance(layer, FiLM1d):
                x = layer(x, y)
            else:
                x = layer(x)
        return x
