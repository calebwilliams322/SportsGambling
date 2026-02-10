"""
Training loop with early stopping and model checkpointing.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from src.config import MODEL_DEFAULTS, MODELS_DIR


class Trainer:
    """
    Handles the training loop, validation, early stopping, and checkpointing.

    Args:
        model: PyTorch model
        learning_rate: optimizer learning rate
        patience: epochs to wait before early stopping
        weight_decay: L2 regularization strength
        use_scheduler: whether to use ReduceLROnPlateau
        scheduler_factor: factor to reduce LR by
        scheduler_patience: epochs before reducing LR
        device: 'cpu', 'cuda', or 'mps'
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = None,
        patience: int = None,
        weight_decay: float = 0.0,
        use_scheduler: bool = False,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
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

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                          weight_decay=weight_decay)
        self.loss_fn = nn.HuberLoss()

        self.scheduler = None
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=scheduler_factor,
                patience=scheduler_patience,
            )

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
            stat_name: str = "model", quiet: bool = False) -> dict:
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

        if not quiet:
            print(f"Training on {self.device} for up to {epochs} epochs...")
            print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Status':>10}")
            print("-" * 50)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Step the LR scheduler if present
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                status = "* saved"
            else:
                patience_counter += 1
                status = f"({patience_counter}/{self.patience})"

            if not quiet:
                print(f"{epoch:>6} | {train_loss:>12.4f} | {val_loss:>12.4f} | {status:>10}")

            # Early stopping
            if patience_counter >= self.patience:
                if not quiet:
                    print(f"\nEarly stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break

        # Load best model
        self.model.load_state_dict(torch.load(best_model_path, weights_only=True))
        if not quiet:
            print(f"Loaded best model from {best_model_path}")

        return {
            "best_val_loss": best_val_loss,
            "best_model_path": str(best_model_path),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "epochs_trained": len(self.train_losses),
        }

    def compute_val_mae(self, val_loader, target_mean: float, target_std: float) -> float:
        """Compute MAE on validation set in original scale."""
        self.model.eval()
        all_preds = []
        all_actuals = []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(self.device)
                preds = self.model(X).cpu().numpy()
                all_preds.append(preds)
                all_actuals.append(y.numpy())
        preds = np.concatenate(all_preds) * target_std + target_mean
        actuals = np.concatenate(all_actuals) * target_std + target_mean
        return float(np.mean(np.abs(preds - actuals)))
