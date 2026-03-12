import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class CNNBackbone(nn.Module):
    def __init__(self, input_channels, cnn_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, cnn_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResBlock1D(cnn_dim),
            ResBlock1D(cnn_dim),
            ResBlock1D(cnn_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, cnn_dim, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(cnn_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]
