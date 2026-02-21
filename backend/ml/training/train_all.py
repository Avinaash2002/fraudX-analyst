"""
FraudX Analyst - Train All Models (FIXED - No Leakage)
========================================================
Runs all three models with proper train/val/test splits.
No data leakage - trustworthy metrics for FYP presentation.
"""

import os, json
from train_xgboost import train_xgboost
from train_lightgbm import train_lightgbm
from train_autoencoder import train_autoencoder

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models_saved')

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  FraudX Analyst — Full Training Pipeline (No Data Leakage)")
    print("=" * 70)

    all_results = {}

    # 1. XGBoost
    _, xgb_meta = train_xgboost()
    all_results["XGBoost"] = xgb_meta

    # 2. LightGBM
    _, lgbm_meta = train_lightgbm()
    all_results["LightGBM"] = lgbm_meta

    # 3. Autoencoder
    _, ae_meta, _ = train_autoencoder()
    all_results["Autoencoder"] = ae_meta

    # Save combined summary
    summary_path = os.path.join(MODELS_DIR, 'all_metrics.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print("\n\n" + "=" * 85)
    print("  TRAINING COMPLETE — Model Comparison (Honest Metrics)")
    print("=" * 85)
    header = f"{'Model':<14} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'ROC-AUC':>8} {'PR-AUC':>8}"
    print(header)
    print("-" * 85)
    for name, m in all_results.items():
        print(f"{name:<14} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1_score']:>8.4f} "
              f"{m['auc_roc']:>8.4f} {m['pr_auc']:>8.4f}")

    print("\n  ✅ All models saved  →", MODELS_DIR)
    print("  ✅ Summary JSON      →", summary_path)
    print("\n  🎯 These are TRUSTWORTHY metrics with:")
    print("     • Proper 70/15/15 train/val/test split")
    print("     • Scalers fit ONLY on training data")
    print("     • Hyperparameters tuned on validation set")
    print("     • Final evaluation done ONCE on test set")
    print("     • No data leakage - FYP presentation ready!")
    print("=" * 85)
