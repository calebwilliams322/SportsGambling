"""
Training loop with early stopping and model checkpointing.
"""
import torch
import torch.nn as nn
from pathlib import Path
from src.config import MODEL_DEFAULTS, MODELS_DIR


class Trainer:
    """
    Handles the training loop, validation, early stopping, and checkpointing.

    Args:
        model: PyTorch model
        learning_rate: optimizer learning rate
        patience: epochs to wait before early stopping
        device: 'cpu', 'cuda', or 'mps'
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = None,
        patience: int = None,
        device: str = None,
    ):
        if learning_rate is None:
            learning_rate = MODEL_DEFAULTS["learning_rate"]
        if patience is None:
            patience = MODEL_DEFAULTS["patience"]
        if device is None:
            device = self._detect_device()

        self.model = model.to(device)
        self.device = device
        self.patience = patience

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.HuberLoss()  # More robust to outliers than MSE

        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(X)
            loss = self.loss_fn(predictions, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self, val_loader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for X, y in val_loader:
            X, y = X.to(self.device), y.to(self.device)
            predictions = self.model(X)
            loss = self.loss_fn(predictions, y)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def fit(self, train_loader, val_loader, epochs: int = None,
            stat_name: str = "model") -> dict:
        """
        Full training loop with early stopping.

        Returns dict with training history and best model path.
        """
        if epochs is None:
            epochs = MODEL_DEFAULTS["epochs"]

        best_val_loss = float("inf")
        patience_counter = 0

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        best_model_path = MODELS_DIR / f"{stat_name}_best.pt"

        print(f"Training on {self.device} for up to {epochs} epochs...")
        print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Status':>10}")
        print("-" * 50)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                status = "* saved"
            else:
                patience_counter += 1
                status = f"({patience_counter}/{self.patience})"

            print(f"{epoch:>6} | {train_loss:>12.4f} | {val_loss:>12.4f} | {status:>10}")

            # Early stopping
            if patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break

        # Load best model
        self.model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print(f"Loaded best model from {best_model_path}")

        return {
            "best_val_loss": best_val_loss,
            "best_model_path": str(best_model_path),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "epochs_trained": len(self.train_losses),
        }
