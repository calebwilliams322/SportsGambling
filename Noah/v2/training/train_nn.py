"""Neural network training with StandardScaler and early stopping."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

from v2.config import (
    NN_BATCH_SIZE, NN_LEARNING_RATE, NN_WEIGHT_DECAY,
    NN_EPOCHS, NN_PATIENCE,
)
from v2.data.dataset import NFLPropsDataset
from v2.models.neural_net import PropsNetV2


def train_nn(X_train, y_train, X_val, y_val, input_dim, verbose=True):
    """
    Train NN and return (model, best_val_loss).
    X_train/X_val should already be scaled.
    """
    train_loader = DataLoader(
        NFLPropsDataset(X_train, y_train),
        batch_size=NN_BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        NFLPropsDataset(X_val, y_val),
        batch_size=256, shuffle=False,
    )

    model = PropsNetV2(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=NN_LEARNING_RATE, weight_decay=NN_WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(NN_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * len(y_batch)
        val_loss /= len(val_loader.dataset)

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.1f}, val_loss={val_loss:.1f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= NN_PATIENCE:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_loss


def predict_nn(model, X):
    """Run inference. X should already be scaled."""
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return np.maximum(preds, 0)
