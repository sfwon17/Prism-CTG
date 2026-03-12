import torch.nn as nn


class VarPredictionHead(nn.Module):
    def __init__(self, embed_dim, num_variables):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_variables),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.predictor(x)


class FeatureLabelHead(nn.Module):
    def __init__(self, embed_dim, n_tokens):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, n_tokens),
        )

    def forward(self, x):
        return self.predictor(x)
