import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import pickle

from config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE,
    CHECKPOINT_DIR, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, PROP_TYPES,
)
from data.features import build_features
from data.dataset import NFLPropsDataset
from models.network import PropsNet


def split_data(df, feature_cols, prop_type):
    train = df[df["season"].isin(TRAIN_SEASONS)]
    val = df[df["season"].isin(VAL_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols].values)
    X_val = scaler.transform(val[feature_cols].values)
    X_test = scaler.transform(test[feature_cols].values)

    y_train = train[prop_type].values
    y_val = val[prop_type].values
    y_test = test[prop_type].values

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def train_one_model(prop_type):
    print(f"\n{'='*60}")
    print(f"Training model for: {prop_type}")
    print(f"{'='*60}")

    # Build features
    print("Building features...")
    df, feature_cols = build_features(prop_type)
    print(f"  Total rows: {len(df)}")

    # Split
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = split_data(
        df, feature_cols, prop_type
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # DataLoaders
    train_loader = DataLoader(
        NFLPropsDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        NFLPropsDataset(X_val, y_val), batch_size=256, shuffle=False
    )

    # Model
    model = PropsNet(input_dim=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{prop_type}_best.pt")

    print("Training...")
    for epoch in range(EPOCHS):
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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.1f}, val_loss={val_loss:.1f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_cols": feature_cols,
                    "prop_type": prop_type,
                    "input_dim": len(feature_cols),
                },
                ckpt_path,
            )
            # Save scaler separately
            with open(os.path.join(CHECKPOINT_DIR, f"{prop_type}_scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Test evaluation
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = PropsNet(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32))
        test_targets = torch.tensor(y_test, dtype=torch.float32)
        mae = torch.abs(test_preds - test_targets).mean().item()
        baseline_mae = torch.abs(test_targets - test_targets.mean()).mean().item()

    print(f"\n  Test MAE: {mae:.1f} yards")
    print(f"  Baseline MAE (predict mean): {baseline_mae:.1f} yards")
    print(f"  Improvement over baseline: {baseline_mae - mae:.1f} yards")
    print(f"  Model saved to {ckpt_path}")

    return mae


def main():
    prop_types = sys.argv[1:] if len(sys.argv) > 1 else list(PROP_TYPES.keys())

    results = {}
    for prop_type in prop_types:
        if prop_type not in PROP_TYPES:
            print(f"Unknown prop type: {prop_type}")
            continue
        mae = train_one_model(prop_type)
        results[prop_type] = mae

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for prop_type, mae in results.items():
        print(f"  {prop_type}: MAE = {mae:.1f} yards")


if __name__ == "__main__":
    main()
