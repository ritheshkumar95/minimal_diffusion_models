import torch
import torch.nn as nn
from functools import partial


class FiLM1d(nn.Module):
    def __init__(self, hidden_dim, n_classes):
        super().__init__()
        self.emb = nn.Embedding(n_classes, hidden_dim * 2)
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False)

    def forward(self, x, y):
        out = self.norm(x)
        alpha, beta = self.emb(y).chunk(2, dim=-1)
        return alpha + out * (1 + beta)


class FiLM2d(nn.Module):
    def __init__(self, hidden_dim, n_classes):
        super().__init__()
        self.emb = nn.Embedding(n_classes, hidden_dim * 2)
        self.norm = nn.BatchNorm2d(hidden_dim, affine=False)

    def forward(self, x, y):
        out = self.norm(x)
        alpha, beta = self.emb(y)[..., None, None].chunk(2, dim=1)
        return alpha + out * (1 + beta)


class MySequential(nn.Sequential):
    def forward(self, x, y):
        for module in self:
            if (
                isinstance(module, FiLM1d)
                or isinstance(module, FiLM2d)
                or isinstance(module, ResBlock)
            ):
                x = module(x, y)
            else:
                x = module(x)
        return x


class ToyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.model = MySequential(
            nn.Linear(input_dim, hidden_dim),
            FiLM1d(hidden_dim, n_classes),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            FiLM1d(hidden_dim, n_classes),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, y):
        return self.model(x, y)


class ResBlock(nn.Module):
    def __init__(self, hidden_dim, n_classes):
        super().__init__()
        self.conv1 = MySequential(
            FiLM2d(hidden_dim, n_classes),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

    def forward(self, x, y):
        return x + self.conv1(x, y)


class MnistNet(nn.Module):
    def __init__(self, input_dim, n_downsample, n_resblocks, ngf, n_classes):
        super().__init__()
        # Add initial layer
        model = [
            nn.Conv2d(input_dim, ngf, kernel_size=7, padding=3),
            FiLM2d(ngf, n_classes),
            nn.ReLU(),
        ]

        # Add downsampling layers
        for i in range(n_downsample):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1
                ),
                FiLM2d(ngf * mult * 2, n_classes),
                nn.ReLU(),
            ]

        # Add ResNet layers
        mult = 2 ** n_downsample
        for i in range(n_resblocks):
            model += [ResBlock(ngf * mult, n_classes)]

        # Add upsampling layers
        for i in range(n_downsample):
            mult = 2 ** (n_downsample - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, ngf * mult // 2, kernel_size=4, stride=2, padding=1
                ),
                FiLM2d(ngf * mult // 2, n_classes),
                nn.ReLU(),
            ]

        # Add output layers
        model += [nn.Conv2d(ngf, input_dim, kernel_size=7, padding=3), nn.Tanh()]

        # Store as sequential layer
        self.model = MySequential(*model)

    def forward(self, x, y):
        return self.model(x, y)
