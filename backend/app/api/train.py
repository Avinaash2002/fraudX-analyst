"""
FraudX Analyst - Train / Models API
=====================================
GET  /api/v1/models            → list all models with metrics
GET  /api/v1/models/compare    → side-by-side comparison data
GET  /api/v1/models/{name}     → single model details
POST /api/v1/train/custom      → upload CSV and train all 3 models
POST /api/v1/train/validate    → validate CSV format before training
"""

import io
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.services.ml_service import get_all_metrics, get_meta, load_all_models
from app.services.training_service import validate_csv, run_full_training

router = APIRouter()


# ── GET /models ────────────────────────────────────────────────────────────────
@router.get("/models")
async def get_models():
    """
    Returns metrics for all three models.
    Used by the Models tab and Home dashboard in Flutter.
    """
    metrics = get_all_metrics()

    response = []
    descriptions = {
        "XGBoost": "Gradient boosted decision trees using scale_pos_weight for class imbalance. Fast and highly accurate for tabular fraud data.",
        "LightGBM": "Light Gradient Boosting Machine. Faster than XGBoost with competitive accuracy. Best performing model in this project.",
        "Autoencoder": "Unsupervised neural network that learns normal transaction patterns. Detects fraud by measuring reconstruction error.",
    }

    for name, meta in metrics.items():
        response.append({
            "model_name"    : name,
            "accuracy"      : meta.get("accuracy",  0),
            "precision"     : meta.get("precision", 0),
            "recall"        : meta.get("recall",    0),
            "f1_score"      : meta.get("f1_score",  0),
            "auc_roc"       : meta.get("auc_roc",   0),
            "algorithm_type": meta.get("algorithm_type", ""),
            "training_time" : meta.get("training_time",  0),
            "description"   : descriptions.get(name, ""),
        })

    # Sort by F1 score descending
    response.sort(key=lambda x: x["f1_score"], reverse=True)
    return {"models": response}


# ── GET /models/compare ────────────────────────────────────────────────────────
@router.get("/models/compare")
async def compare_models():
    """
    Returns structured comparison data for bar charts in the Flutter Models tab.
    """
    metrics = get_all_metrics()
    model_names = list(metrics.keys())

    def extract_metric(metric_key):
        return [round(metrics[name].get(metric_key, 0) * 100, 2) for name in model_names]

    return {
        "models"   : model_names,
        "metrics"  : {
            "accuracy" : extract_metric("accuracy"),
            "precision": extract_metric("precision"),
            "recall"   : extract_metric("recall"),
            "f1_score" : extract_metric("f1_score"),
            "auc_roc"  : extract_metric("auc_roc"),
        }
    }


# ── GET /models/{name} ─────────────────────────────────────────────────────────
@router.get("/models/{model_name}")
async def get_model_detail(model_name: str):
    """
    Returns full details for a single model including best params.
    """
    valid = ["XGBoost", "LightGBM", "Autoencoder"]
    if model_name not in valid:
        raise HTTPException(status_code=404, detail=f"Model not found. Choose from {valid}")

    meta = get_meta(model_name)
    if not meta:
        raise HTTPException(status_code=404, detail="Model metrics not found")

    return meta


# ── POST /train/validate ───────────────────────────────────────────────────────
@router.post("/train/validate")
async def validate_dataset(file: UploadFile = File(...)):
    """
    Validates an uploaded CSV file before training.
    Returns dataset stats (rows, fraud count, etc.) or error message.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        result = validate_csv(df)

        if not result["valid"]:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "valid": True,
            "filename": file.filename,
            "rows": result["rows"],
            "columns": result["columns"],
            "fraud_count": result["fraud_count"],
            "normal_count": result["normal_count"],
            "imbalance_ratio": result["imbalance_ratio"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")


# ── POST /train/custom ─────────────────────────────────────────────────────────
@router.post("/train/custom")
async def train_custom_dataset(file: UploadFile = File(...)):
    """
    Uploads a CSV file and trains all 3 models (XGBoost, LightGBM, Autoencoder).
    Uses reduced Optuna trials (10) for faster execution (~3-5 min).
    Returns real metrics for all models.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Validate
        validation = validate_csv(df)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["error"])

        print(f"\n  📊 Training on custom dataset: {file.filename}")
        print(f"     Rows: {len(df):,} | Fraud: {validation['fraud_count']:,} | Normal: {validation['normal_count']:,}")

        # Run full training pipeline (10 Optuna trials for speed)
        result = run_full_training(df, n_trials=10)

        # Reload models only if any were upgraded
        if result.get("upgraded"):
            load_all_models()
            print(f"  🔄 Models reloaded: {', '.join(result['upgraded'])}")
        else:
            print(f"  ✅ No models upgraded — keeping existing models")

        return {
            "success": True,
            "filename": file.filename,
            "dataset_rows": result["dataset_rows"],
            "best_model": result["best_model"],
            "upgraded": result.get("upgraded", []),
            "kept": result.get("kept", []),
            "results": {
                name: {
                    "model_name": metrics["model_name"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "auc_roc": metrics["auc_roc"],
                    "training_time": metrics["training_time"],
                    "algorithm_type": metrics["algorithm_type"],
                    "upgraded": metrics.get("upgraded", False),
                    "existing_f1": metrics.get("existing_f1", 0),
                }
                for name, metrics in result["results"].items()
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# ── GET /sample-transaction ────────────────────────────────────────────────────
import os, random
import numpy as np
from sklearn.model_selection import train_test_split

_test_set_cache = None

def _load_test_set():
    """
    Loads the dataset and extracts ONLY the test set (15%).
    Uses the same split logic as preprocess.py (random_state=42, stratify)
    so these are transactions the model has NEVER seen during training.
    """
    global _test_set_cache
    if _test_set_cache is not None:
        return _test_set_cache

    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'creditcard.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ml', 'data', 'creditcard.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ml', 'training', '..', 'data', 'creditcard.csv')

    df = pd.read_csv(csv_path)
    y = df['Class'].values

    # Replicate EXACT same split as preprocess.py
    _, X_temp, _, y_temp = train_test_split(
        df, y, test_size=0.30, random_state=42, stratify=y
    )
    _, X_test, _, _ = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    _test_set_cache = X_test
    print(f"  📊 Test set loaded: {len(X_test):,} transactions "
          f"(fraud: {(X_test['Class']==1).sum()}, normal: {(X_test['Class']==0).sum()})")
    return _test_set_cache


@router.get("/sample-transaction")
async def get_sample_transaction(type: str = "random"):
    """
    Returns a real transaction from the HELD-OUT TEST SET (15% of dataset).
    These are transactions the model has NEVER seen during training.
    This proves the model generalizes to unseen data — not memorization.

    Query params:
        type: 'fraud' | 'normal' | 'random'
    """
    try:
        df = _load_test_set()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load test set: {e}")

    if type == "fraud":
        subset = df[df['Class'] == 1]
    elif type == "normal":
        subset = df[df['Class'] == 0]
    else:
        subset = df

    if subset.empty:
        raise HTTPException(status_code=404, detail=f"No {type} transactions found in test set")

    # Pick a random row from the test set
    row = subset.sample(n=1).iloc[0]

    return {
        "type": "FRAUD" if row['Class'] == 1 else "NORMAL",
        "source": "held-out test set (never seen during training)",
        "amount": round(float(row['Amount']), 2),
        "time": round(float(row['Time']), 2),
        "v1": round(float(row['V1']), 6),
        "v2": round(float(row['V2']), 6),
        "v3": round(float(row['V3']), 6),
        "v4": round(float(row['V4']), 6),
        "v5": round(float(row['V5']), 6),
        "v6": round(float(row['V6']), 6),
        "v7": round(float(row['V7']), 6),
        "v8": round(float(row['V8']), 6),
        "v9": round(float(row['V9']), 6),
        "v10": round(float(row['V10']), 6),
        "v11": round(float(row['V11']), 6),
        "v12": round(float(row['V12']), 6),
        "v13": round(float(row['V13']), 6),
        "v14": round(float(row['V14']), 6),
        "v15": round(float(row['V15']), 6),
        "v16": round(float(row['V16']), 6),
        "v17": round(float(row['V17']), 6),
        "v18": round(float(row['V18']), 6),
        "v19": round(float(row['V19']), 6),
        "v20": round(float(row['V20']), 6),
        "v21": round(float(row['V21']), 6),
        "v22": round(float(row['V22']), 6),
        "v23": round(float(row['V23']), 6),
        "v24": round(float(row['V24']), 6),
        "v25": round(float(row['V25']), 6),
        "v26": round(float(row['V26']), 6),
        "v27": round(float(row['V27']), 6),
        "v28": round(float(row['V28']), 6),
    }
