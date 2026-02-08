#!/usr/bin/env python3
"""
Train a player props model for a specific stat.

Usage:
    python scripts/train.py --stat passing_yards
    python scripts/train.py --stat rushing_yards
    python scripts/train.py --stat receiving_yards
    python scripts/train.py --all
"""
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TARGET_STATS, MODELS_DIR
from src.models.player_props import PlayerPropsNet
from src.training.dataset import prepare_data
from src.training.trainer import Trainer
from src.training.evaluate import evaluate_model, compare_to_baseline


def train_single_stat(stat_name: str):
    """Train a model for one stat type."""
    print(f"\n{'#' * 60}")
    print(f"Training model for: {stat_name}")
    print(f"{'#' * 60}\n")

    # Prepare data
    train_loader, val_loader, test_loader, scaler, feature_cols, target_info = prepare_data(stat_name)

    # Create model
    input_size = len(feature_cols)
    model = PlayerPropsNet(input_size=input_size)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Train
    trainer = Trainer(model)
    history = trainer.fit(train_loader, val_loader, stat_name=stat_name)

    # Evaluate on test set (denormalize predictions back to original scale)
    results = evaluate_model(model, test_loader, device=trainer.device,
                             stat_name=stat_name, target_stats=target_info)
    baseline = compare_to_baseline(results["predictions"], results["actuals"], stat_name)

    # Save metadata
    meta = {
        "stat": stat_name,
        "n_features": input_size,
        "feature_columns": feature_cols,
        "target_normalization": target_info,
        "history": {
            "best_val_loss": history["best_val_loss"],
            "epochs_trained": history["epochs_trained"],
        },
        "test_metrics": results["metrics"],
        "baseline_comparison": baseline,
    }

    meta_path = MODELS_DIR / f"{stat_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nSaved metadata to {meta_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train player props model")
    parser.add_argument("--stat", type=str, help="Stat to predict (e.g. passing_yards)")
    parser.add_argument("--all", action="store_true", help="Train models for all stats")
    args = parser.parse_args()

    if args.all:
        for stat in TARGET_STATS:
            train_single_stat(stat)
    elif args.stat:
        if args.stat not in TARGET_STATS:
            print(f"Unknown stat: {args.stat}")
            print(f"Available: {TARGET_STATS}")
            sys.exit(1)
        train_single_stat(args.stat)
    else:
        print("Specify --stat <name> or --all")
        print(f"Available stats: {TARGET_STATS}")
        sys.exit(1)


if __name__ == "__main__":
    main()
