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


class ToyPosterior(nn.Module):
    def __init__(self, input_dim, hidden_dim, enc_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, enc_dim * 2),
        )

    def forward(self, x, temp=1.0):
        mu, logs = self.model(x).chunk(2, dim=-1)
        z = mu + torch.randn_like(logs) * logs.exp() * temp
        return z, mu, logs


class ToyDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, enc_dim, n_classes):
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Linear(input_dim + enc_dim, hidden_dim),
                FiLM1d(hidden_dim, n_classes),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                FiLM1d(hidden_dim, n_classes),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            ]
        )

    def forward(self, x, z, y):
        x = torch.cat([x, z], dim=-1)
        for layer in self.model:
            if isinstance(layer, FiLM1d):
                x = layer(x, y)
            else:
                x = layer(x)
        return x


class ToyVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, enc_dim, n_classes):
        super().__init__()
        self.posterior = ToyPosterior(input_dim, hidden_dim, enc_dim)
        self.decoder = ToyDecoder(input_dim, hidden_dim, enc_dim, n_classes)

    def forward(self, x_0, x_t, y):
        z_q, mu, logs = self.posterior(x_0)
        pred_z = self.decoder(x_t, z_q, y)
        return pred_z, mu, logs

    def generate(self, z_p, x_t, y):
        return self.decoder(x_t, z_p, y)
