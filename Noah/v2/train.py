"""
V2 main entry point — trains the full pipeline for all prop types.

Usage:
    python v2/train.py                       # train all 3 prop types
    python v2/train.py passing_yards         # train one specific prop type
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v2.config import PROP_TYPES
from v2.training.train_ensemble import train_prop_type


def main():
    prop_types = sys.argv[1:] if len(sys.argv) > 1 else list(PROP_TYPES.keys())

    results = {}
    for prop_type in prop_types:
        if prop_type not in PROP_TYPES:
            print(f"Unknown prop type: {prop_type}")
            continue
        result = train_prop_type(prop_type)
        results[prop_type] = result

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("V2 TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Prop Type':<20} {'NN MAE':>8} {'GBM MAE':>9} {'Ensemble':>10} {'Baseline':>10} {'Features':>9}")
    print(f"  {'-'*66}")
    for prop_type, r in results.items():
        print(
            f"  {prop_type:<20} {r['nn_mae']:>7.1f} {r['gbm_mae']:>8.1f} "
            f"{r['ensemble_mae']:>9.1f} {r['baseline_mae']:>9.1f} {r['n_features']:>8d}"
        )
    print(f"\n  Ensemble weights:")
    for prop_type, r in results.items():
        print(f"    {prop_type}: NN={r['nn_weight']:.2f}, GBM={r['gbm_weight']:.2f}")

    print("\nDone. Run 'python v2/predict_sb60.py' for SB60 predictions.")


if __name__ == "__main__":
    main()
