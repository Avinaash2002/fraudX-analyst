"""
FraudX Analyst - Training Service
====================================
Handles full training pipeline for custom CSV uploads.
Uses reduced Optuna trials (10) for faster execution (~3-5 min).
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import tempfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score,
                              confusion_matrix)

import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
tf.get_logger().setLevel('ERROR')

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'ml', 'training', 'models_saved')


# ── Validation ─────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]

def _load_existing_f1(model_name: str) -> float:
    """Load existing F1 score for a model. Returns 0 if not found."""
    name_map = {"XGBoost": "xgboost", "LightGBM": "lightgbm", "Autoencoder": "autoencoder"}
    metrics_file = os.path.join(MODELS_DIR, f'{name_map.get(model_name, model_name.lower())}_metrics.json')
    try:
        with open(metrics_file) as f:
            return json.load(f).get('f1_score', 0)
    except Exception:
        return 0

def validate_csv(df: pd.DataFrame) -> dict:
    """Validate uploaded CSV has correct format."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return {"valid": False, "error": f"Missing columns: {', '.join(missing[:5])}"}

    if df['Class'].nunique() < 2:
        return {"valid": False, "error": "Dataset must contain both fraud (1) and normal (0) classes"}

    if len(df) < 100:
        return {"valid": False, "error": "Dataset too small (minimum 100 rows)"}

    fraud_count = int((df['Class'] == 1).sum())
    normal_count = int((df['Class'] == 0).sum())

    return {
        "valid": True,
        "rows": len(df),
        "columns": len(df.columns),
        "fraud_count": fraud_count,
        "normal_count": normal_count,
        "imbalance_ratio": round(fraud_count / len(df), 5),
    }


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame):
    """Preprocess with proper 70/15/15 split, no leakage."""
    feature_names = [c for c in df.columns if c != 'Class']
    X = df[feature_names].copy()
    y = df['Class'].values

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Scale (fit on train only)
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()

    amount_scaler.fit(X_train[['Amount']])
    time_scaler.fit(X_train[['Time']])

    for split in [X_train, X_val, X_test]:
        split.loc[:, 'Amount'] = amount_scaler.transform(split[['Amount']])
        split.loc[:, 'Time'] = time_scaler.transform(split[['Time']])

    # Save scalers
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(amount_scaler, os.path.join(MODELS_DIR, 'amount_scaler.pkl'))
    joblib.dump(time_scaler, os.path.join(MODELS_DIR, 'time_scaler.pkl'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.pkl'))

    return (X_train.values, X_val.values, X_test.values,
            y_train, y_val, y_test, feature_names)


# ── Metrics helper ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


# ── Train XGBoost ──────────────────────────────────────────────────────────────
def train_xgboost_fast(X_train, X_val, X_test, y_train, y_val, y_test, n_trials=10):
    print("  🔷 Training XGBoost (fast mode)…")
    t0 = time.time()

    n_normal = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    spw = round(n_normal / n_fraud, 2)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": spw,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["scale_pos_weight"] = spw
    best_params["eval_metric"] = "logloss"
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    training_time = round(time.time() - t0, 2)
    metrics["training_time"] = training_time
    metrics["model_name"] = "XGBoost"
    metrics["algorithm_type"] = "supervised"

    # Save only if F1 is better than existing
    existing_f1 = _load_existing_f1("XGBoost")
    if metrics["f1_score"] > existing_f1:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
        import shap
        explainer = shap.TreeExplainer(model)
        joblib.dump(explainer, os.path.join(MODELS_DIR, 'xgboost_explainer.pkl'))
        with open(os.path.join(MODELS_DIR, 'xgboost_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        metrics["upgraded"] = True
        print(f"  ✅ XGBoost UPGRADED: F1={metrics['f1_score']:.4f} > {existing_f1:.4f}")
    else:
        metrics["upgraded"] = False
        print(f"  ⏭️  XGBoost KEPT: new F1={metrics['f1_score']:.4f} ≤ existing F1={existing_f1:.4f}")
    metrics["existing_f1"] = existing_f1
    return metrics


# ── Train LightGBM ─────────────────────────────────────────────────────────────
def train_lightgbm_fast(X_train, X_val, X_test, y_train, y_val, y_test, n_trials=10):
    print("  🔶 Training LightGBM (fast mode)…")
    t0 = time.time()

    n_normal = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    cw = {0: 1, 1: round(n_normal / n_fraud, 2)}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "class_weight": cw,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["class_weight"] = cw
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["verbose"] = -1

    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    training_time = round(time.time() - t0, 2)
    metrics["training_time"] = training_time
    metrics["model_name"] = "LightGBM"
    metrics["algorithm_type"] = "supervised"

    # Save only if F1 is better than existing
    existing_f1 = _load_existing_f1("LightGBM")
    if metrics["f1_score"] > existing_f1:
        joblib.dump(model, os.path.join(MODELS_DIR, 'lightgbm_model.pkl'))
        import shap
        explainer = shap.TreeExplainer(model)
        joblib.dump(explainer, os.path.join(MODELS_DIR, 'lightgbm_explainer.pkl'))
        with open(os.path.join(MODELS_DIR, 'lightgbm_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        metrics["upgraded"] = True
        print(f"  ✅ LightGBM UPGRADED: F1={metrics['f1_score']:.4f} > {existing_f1:.4f}")
    else:
        metrics["upgraded"] = False
        print(f"  ⏭️  LightGBM KEPT: new F1={metrics['f1_score']:.4f} ≤ existing F1={existing_f1:.4f}")
    metrics["existing_f1"] = existing_f1
    return metrics


# ── Train Autoencoder ──────────────────────────────────────────────────────────
def train_autoencoder_fast(X_train, X_val, X_test, y_train, y_val, y_test):
    print("  🟣 Training Autoencoder (fast mode)…")
    t0 = time.time()

    # Normal-only training data
    X_normal = X_train[y_train == 0]

    input_dim = X_train.shape[1]
    inputs = Input(shape=(input_dim,))
    x = Dense(32, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    decoded = Dense(input_dim, activation='linear')(x)

    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')

    model.fit(
        X_normal, X_normal,
        epochs=30,
        batch_size=256,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
        verbose=0
    )

    # Compute reconstruction errors
    def compute_errors(X):
        recon = model.predict(X, verbose=0)
        mse = np.mean(np.power(X - recon, 2), axis=1)
        mae = np.mean(np.abs(X - recon), axis=1)
        max_err = np.max(np.abs(X - recon), axis=1)
        return 0.5 * mse + 0.3 * mae + 0.2 * max_err

    # Optimize threshold on validation
    errors_val = compute_errors(X_val)
    best_f1, best_threshold = 0, 0
    for thresh in np.linspace(errors_val.min(), errors_val.max(), 200):
        y_pred = (errors_val > thresh).astype(int)
        if y_pred.sum() == 0:
            continue
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    # Evaluate on test
    errors_test = compute_errors(X_test)
    y_pred = (errors_test > best_threshold).astype(int)
    y_prob = (errors_test - errors_test.min()) / (errors_test.max() - errors_test.min() + 1e-10)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    training_time = round(time.time() - t0, 2)
    metrics["training_time"] = training_time
    metrics["model_name"] = "Autoencoder"
    metrics["algorithm_type"] = "unsupervised"
    metrics["threshold"] = float(best_threshold)

    # Save only if F1 is better than existing
    existing_f1 = _load_existing_f1("Autoencoder")
    if metrics["f1_score"] > existing_f1:
        model.save(os.path.join(MODELS_DIR, 'autoencoder_model.keras'))
        with open(os.path.join(MODELS_DIR, 'autoencoder_threshold.json'), 'w') as f:
            json.dump({"threshold": float(best_threshold)}, f, indent=2)
        with open(os.path.join(MODELS_DIR, 'autoencoder_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        metrics["upgraded"] = True
        print(f"  ✅ Autoencoder UPGRADED: F1={metrics['f1_score']:.4f} > {existing_f1:.4f}")
    else:
        metrics["upgraded"] = False
        print(f"  ⏭️  Autoencoder KEPT: new F1={metrics['f1_score']:.4f} ≤ existing F1={existing_f1:.4f}")
    metrics["existing_f1"] = existing_f1
    return metrics


# ── Full Pipeline ──────────────────────────────────────────────────────────────
def run_full_training(df: pd.DataFrame, n_trials: int = 10) -> dict:
    """
    Run the complete training pipeline on a custom dataset.
    Returns metrics for all 3 models.
    """
    print("\n" + "=" * 60)
    print("  FraudX — Custom Dataset Training (Fast Mode)")
    print("=" * 60)

    # Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_dataframe(df)
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Train all 3
    xgb_metrics = train_xgboost_fast(X_train, X_val, X_test, y_train, y_val, y_test, n_trials)
    lgbm_metrics = train_lightgbm_fast(X_train, X_val, X_test, y_train, y_val, y_test, n_trials)
    ae_metrics = train_autoencoder_fast(X_train, X_val, X_test, y_train, y_val, y_test)

    new_metrics = {
        "XGBoost": xgb_metrics,
        "LightGBM": lgbm_metrics,
        "Autoencoder": ae_metrics,
    }

    # Load existing all_metrics.json and only update upgraded models
    all_metrics_path = os.path.join(MODELS_DIR, 'all_metrics.json')
    try:
        with open(all_metrics_path) as f:
            existing_all = json.load(f)
    except Exception:
        existing_all = {}

    for name, metrics in new_metrics.items():
        if metrics.get("upgraded", False):
            save_metrics = {k: v for k, v in metrics.items() if k not in ("upgraded", "existing_f1")}
            existing_all[name] = save_metrics

    with open(all_metrics_path, 'w') as f:
        json.dump(existing_all, f, indent=2)

    upgraded = [n for n, m in new_metrics.items() if m.get("upgraded")]
    kept = [n for n, m in new_metrics.items() if not m.get("upgraded")]
    best_name = max(existing_all, key=lambda k: existing_all[k].get('f1_score', 0)) if existing_all else "LightGBM"

    print(f"\n  🏆 Best Active Model: {best_name}")
    if upgraded: print(f"  ⬆️  Upgraded: {', '.join(upgraded)}")
    if kept: print(f"  ✅ Kept: {', '.join(kept)}")

    return {
        "success": True,
        "best_model": best_name,
        "results": new_metrics,
        "dataset_rows": len(df),
        "upgraded": upgraded,
        "kept": kept,
    }
