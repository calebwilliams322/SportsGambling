#!/usr/bin/env python3
"""
Hyperparameter sweep — finds the best model config for each stat.

Tunes on validation MAE (original scale), then evaluates best config on test set.
"""
import sys
import json
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR
from src.models.player_props import PlayerPropsNet
from src.training.dataset import prepare_data
from src.training.trainer import Trainer
from src.training.evaluate import evaluate_model, compare_to_baseline

# --- Hyperparameter grid ---
SWEEP_GRID = {
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size": [32, 64, 128],
    "hidden_sizes": [
        [128, 64, 32],
        [256, 128, 64],
        [256, 128, 64, 32],
        [512, 256, 128],
    ],
    "dropout": [0.2, 0.3, 0.4],
    "weight_decay": [0, 1e-5, 1e-4],
    "use_scheduler": [True, False],
    "patience": [15, 20],
}

# Total combinations would be 3*3*4*3*3*2*2 = 1296 — way too many.
# Instead, do a staged search: first sweep architecture + LR, then fine-tune.

STAGE_1 = {
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "hidden_sizes": [
        [128, 64, 32],
        [256, 128, 64],
        [256, 128, 64, 32],
        [512, 256, 128],
    ],
    "dropout": [0.2, 0.3],
    "use_scheduler": [True, False],
}
# Fixed for stage 1
STAGE_1_FIXED = {"batch_size": 64, "weight_decay": 0, "patience": 20}

STAGE_2_EXTRAS = {
    "batch_size": [32, 64, 128],
    "weight_decay": [0, 1e-5, 1e-4],
    "patience": [15, 20, 30],
}


def run_single_config(stat_name, config, feature_cols, train_loader, val_loader,
                      target_info, device=None):
    """Train one config and return val MAE."""
    model = PlayerPropsNet(
        input_size=len(feature_cols),
        hidden_sizes=config["hidden_sizes"],
        dropout=config["dropout"],
    )
    trainer = Trainer(
        model,
        learning_rate=config["learning_rate"],
        patience=config["patience"],
        weight_decay=config.get("weight_decay", 0),
        use_scheduler=config.get("use_scheduler", False),
        scheduler_patience=config.get("scheduler_patience", 5),
        device=device,
    )
    trainer.fit(train_loader, val_loader, stat_name=stat_name, quiet=True)
    val_mae = trainer.compute_val_mae(val_loader, target_info["mean"], target_info["std"])
    epochs = len(trainer.train_losses)
    return val_mae, epochs


def sweep_stat(stat_name):
    """Run full hyperparameter sweep for one stat."""
    print(f"\n{'#' * 60}")
    print(f"  HYPERPARAMETER SWEEP: {stat_name}")
    print(f"{'#' * 60}\n")

    # Prepare data with default batch size first
    train_loader, val_loader, test_loader, scaler, feature_cols, target_info = \
        prepare_data(stat_name)

    # For batch size changes, we need to re-create loaders
    from src.training.dataset import prepare_data as _prepare_data

    # --- Stage 1: Architecture + LR sweep ---
    print("Stage 1: Architecture + Learning Rate sweep")
    print(f"{'#':>4} | {'LR':>8} | {'Hidden':>20} | {'Drop':>5} | {'Sched':>5} | {'ValMAE':>8} | {'Epochs':>6}")
    print("-" * 75)

    stage1_results = []
    configs = list(itertools.product(
        STAGE_1["learning_rate"],
        STAGE_1["hidden_sizes"],
        STAGE_1["dropout"],
        STAGE_1["use_scheduler"],
    ))

    for i, (lr, hidden, drop, sched) in enumerate(configs, 1):
        config = {
            "learning_rate": lr,
            "hidden_sizes": hidden,
            "dropout": drop,
            "use_scheduler": sched,
            "batch_size": STAGE_1_FIXED["batch_size"],
            "weight_decay": STAGE_1_FIXED["weight_decay"],
            "patience": STAGE_1_FIXED["patience"],
        }
        val_mae, epochs = run_single_config(
            stat_name, config, feature_cols, train_loader, val_loader, target_info
        )
        hidden_str = "x".join(map(str, hidden))
        print(f"{i:>4} | {lr:>8.0e} | {hidden_str:>20} | {drop:>5.1f} | {str(sched):>5} | {val_mae:>8.2f} | {epochs:>6}")
        stage1_results.append((val_mae, config))

    # Sort by val MAE
    stage1_results.sort(key=lambda x: x[0])
    best_s1_mae, best_s1_config = stage1_results[0]
    print(f"\nStage 1 best: MAE={best_s1_mae:.2f}")
    print(f"  Config: lr={best_s1_config['learning_rate']}, "
          f"hidden={best_s1_config['hidden_sizes']}, "
          f"dropout={best_s1_config['dropout']}, "
          f"scheduler={best_s1_config['use_scheduler']}")

    # --- Stage 2: Fine-tune batch size, weight decay, patience ---
    print(f"\nStage 2: Fine-tuning batch size, weight decay, patience")
    print(f"{'#':>4} | {'BS':>4} | {'WD':>8} | {'Pat':>4} | {'ValMAE':>8} | {'Epochs':>6}")
    print("-" * 50)

    stage2_results = []
    combos = list(itertools.product(
        STAGE_2_EXTRAS["batch_size"],
        STAGE_2_EXTRAS["weight_decay"],
        STAGE_2_EXTRAS["patience"],
    ))

    for i, (bs, wd, pat) in enumerate(combos, 1):
        # Re-prepare data with different batch size if needed
        if bs != 64:
            from src.config import MODEL_DEFAULTS
            old_bs = MODEL_DEFAULTS["batch_size"]
            MODEL_DEFAULTS["batch_size"] = bs
            tl, vl, tstl, sc, fc, ti = _prepare_data(stat_name)
            MODEL_DEFAULTS["batch_size"] = old_bs
        else:
            tl, vl, tstl, sc, fc, ti = train_loader, val_loader, test_loader, scaler, feature_cols, target_info

        config = {**best_s1_config, "batch_size": bs, "weight_decay": wd, "patience": pat}
        val_mae, epochs = run_single_config(stat_name, config, fc, tl, vl, ti)
        print(f"{i:>4} | {bs:>4} | {wd:>8.0e} | {pat:>4} | {val_mae:>8.2f} | {epochs:>6}")
        stage2_results.append((val_mae, config))

    stage2_results.sort(key=lambda x: x[0])
    best_mae, best_config = stage2_results[0]

    # Compare to stage 1 best
    if best_mae >= best_s1_mae:
        best_mae = best_s1_mae
        best_config = best_s1_config
        print(f"\nStage 2 didn't improve. Keeping Stage 1 best.")
    else:
        print(f"\nStage 2 best: MAE={best_mae:.2f}")

    print(f"\n{'=' * 60}")
    print(f"  BEST CONFIG for {stat_name}: Val MAE = {best_mae:.2f}")
    print(f"{'=' * 60}")
    for k, v in sorted(best_config.items()):
        print(f"  {k}: {v}")

    # --- Final: retrain with best config and evaluate on test ---
    print(f"\nRetraining final model with best config...")
    if best_config["batch_size"] != 64:
        from src.config import MODEL_DEFAULTS
        old_bs = MODEL_DEFAULTS["batch_size"]
        MODEL_DEFAULTS["batch_size"] = best_config["batch_size"]
        train_loader, val_loader, test_loader, scaler, feature_cols, target_info = \
            _prepare_data(stat_name)
        MODEL_DEFAULTS["batch_size"] = old_bs

    model = PlayerPropsNet(
        input_size=len(feature_cols),
        hidden_sizes=best_config["hidden_sizes"],
        dropout=best_config["dropout"],
    )
    trainer = Trainer(
        model,
        learning_rate=best_config["learning_rate"],
        patience=best_config["patience"],
        weight_decay=best_config.get("weight_decay", 0),
        use_scheduler=best_config.get("use_scheduler", False),
    )
    history = trainer.fit(train_loader, val_loader, stat_name=stat_name, quiet=False)

    results = evaluate_model(model, test_loader, device=trainer.device,
                             stat_name=stat_name, target_stats=target_info)
    baseline = compare_to_baseline(results["predictions"], results["actuals"], stat_name)

    # Save metadata
    meta = {
        "stat": stat_name,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "target_normalization": target_info,
        "best_config": {k: str(v) if not isinstance(v, (int, float, bool, list)) else v
                        for k, v in best_config.items()},
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

    return best_config, best_mae


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", type=str, required=True)
    args = parser.parse_args()

    sweep_stat(args.stat)
