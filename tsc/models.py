"""Arquiteturas de deep learning (PyTorch) para classificação binária de séries temporais.

Todos os modelos recebem tensores com shape (batch, window_size, n_features) e retornam
logits com shape (batch,). Use `torch.sigmoid` para obter a probabilidade de falha.
"""

import warnings

import torch
from torch import nn

# padding="same" com kernels pares (InceptionTime/ResNet1D) funciona, mas o PyTorch emite um aviso
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths")


def _conv_relu_bn(in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size, padding="same"),
        nn.ReLU(),
        nn.BatchNorm1d(out_ch),
    )


def _head(in_features: int, dropout: float, hidden: int | None = 64) -> nn.Sequential:
    """Global average pooling + camadas densas até um único logit."""
    pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
    if hidden is None:
        return nn.Sequential(pool, nn.Dropout(dropout), nn.Linear(in_features, 1))
    return nn.Sequential(
        pool, nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
    )


class CNN1D(nn.Module):
    """Rede convolucional 1D com 4 blocos Conv-ReLU-BatchNorm."""

    def __init__(self, n_features: int):
        super().__init__()
        self.features = nn.Sequential(
            _conv_relu_bn(n_features, 64, 7),
            nn.Dropout(0.1),
            _conv_relu_bn(64, 128, 5),
            nn.MaxPool1d(2),
            _conv_relu_bn(128, 256, 3),
            nn.Dropout(0.1),
            _conv_relu_bn(256, 128, 3),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
        )
        self.head = _head(128, dropout=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.transpose(1, 2))).squeeze(-1)


class InceptionModule(nn.Module):
    """Bloco Inception 1D: convoluções paralelas com kernels de tamanhos diferentes."""

    def __init__(
        self,
        in_ch: int,
        n_filters: int = 32,
        kernel_sizes: tuple[int, ...] = (10, 20, 40),
        bottleneck_size: int = 32,
    ):
        super().__init__()
        use_bottleneck = bottleneck_size and in_ch > 1
        self.bottleneck = (
            nn.Sequential(nn.Conv1d(in_ch, bottleneck_size, 1), nn.ReLU())
            if use_bottleneck
            else nn.Identity()
        )
        branch_ch = bottleneck_size if use_bottleneck else in_ch
        self.convs = nn.ModuleList(
            nn.Sequential(nn.Conv1d(branch_ch, n_filters, k, padding="same"), nn.ReLU())
            for k in kernel_sizes
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_ch, n_filters, 1), nn.ReLU()
        )
        self.out_channels = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        out = torch.cat([conv(z) for conv in self.convs] + [self.pool_branch(x)], dim=1)
        return torch.relu(self.bn(out))


class InceptionTime(nn.Module):
    """InceptionTime simplificado: `depth` módulos Inception empilhados."""

    def __init__(self, n_features: int, depth: int = 3, n_filters: int = 32):
        super().__init__()
        modules, in_ch = [], n_features
        for _ in range(depth):
            module = InceptionModule(in_ch, n_filters=n_filters)
            modules.append(module)
            in_ch = module.out_channels
        self.features = nn.Sequential(*modules)
        self.head = _head(in_ch, dropout=0.25, hidden=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.transpose(1, 2))).squeeze(-1)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding="same"),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding="same"),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    """ResNet 1D com 3 blocos residuais."""

    def __init__(self, n_features: int):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock1D(n_features, 64, 8),
            ResidualBlock1D(64, 128, 5),
            ResidualBlock1D(128, 128, 3),
        )
        self.head = _head(128, dropout=0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.transpose(1, 2))).squeeze(-1)


class ResidualBlock2D(nn.Module):
    """Bloco residual com convoluções (k, 1): filtra no tempo, variável a variável."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (7, 1), padding="same"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, (5, 1), padding="same"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, (3, 1), padding="same"),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.shortcut(x))


class ResNet2D(nn.Module):
    """ResNet 2D que trata a janela (tempo x variáveis) como uma imagem de 1 canal."""

    def __init__(self, n_features: int):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock2D(1, 64), ResidualBlock2D(64, 128), ResidualBlock2D(128, 128)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.unsqueeze(1))).squeeze(-1)


MODELS = {
    "CNN1D": CNN1D,
    "InceptionTime": InceptionTime,
    "ResNet1D": ResNet1D,
    "ResNet2D": ResNet2D,
}
