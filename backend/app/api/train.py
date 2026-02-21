"""
FraudX Analyst - Train / Models API
=====================================
GET /api/v1/models         → list all models with metrics
GET /api/v1/models/compare → side-by-side comparison data
GET /api/v1/models/{name}  → single model details
"""

from fastapi import APIRouter, HTTPException
from app.services.ml_service import get_all_metrics, get_meta

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
