import torch.nn as nn

from v2.config import NN_HIDDEN_DIMS, NN_DROPOUT


class PropsNetV2(nn.Module):
    """Deeper feed-forward net: 128→64→32 with LayerNorm and higher dropout."""

    def __init__(self, input_dim, hidden_dims=NN_HIDDEN_DIMS, dropout=NN_DROPOUT):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
