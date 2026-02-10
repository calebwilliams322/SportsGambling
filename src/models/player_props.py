"""
PyTorch model definitions for player prop predictions.

MVP: Feed-forward neural network for regression.
Future: LSTM, Transformer, etc.
"""
import torch
import torch.nn as nn
from src.config import MODEL_DEFAULTS


class PlayerPropsNet(nn.Module):
    """
    Feed-forward neural network for predicting a single stat value.

    Architecture:
        Input → [Hidden + ReLU + Dropout] x N → Output (1 value)

    Args:
        input_size: number of input features
        hidden_sizes: list of hidden layer sizes (default: [128, 64, 32])
        dropout: dropout rate between layers (default: 0.3)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = None,
        dropout: float = None,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = MODEL_DEFAULTS["hidden_sizes"]
        if dropout is None:
            dropout = MODEL_DEFAULTS["dropout"]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer: single regression value
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
