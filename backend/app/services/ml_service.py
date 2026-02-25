"""
FraudX Analyst - ML Service
=============================
Loads all trained models once at startup and keeps them in memory.
Runs inference for XGBoost, LightGBM, and Autoencoder.
"""

import os, json, time
import numpy as np
import joblib
import tensorflow as tf
from typing import Tuple

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'ml','training','models_saved')

# ── Global model cache (loaded once at startup) ────────────────────────────────
_models     = {}
_explainers = {}
_scalers    = {}
_meta       = {}


def load_all_models():
    """
    Called once when the FastAPI server starts.
    Loads all models into memory so predictions are fast.
    """
    global _models, _explainers, _scalers, _meta

    print("  Loading ML models …")

    # Scalers
    _scalers["amount"] = joblib.load(os.path.join(MODELS_DIR, "amount_scaler.pkl"))
    _scalers["time"]   = joblib.load(os.path.join(MODELS_DIR, "time_scaler.pkl"))
    _scalers["features"] = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

    # XGBoost
    _models["XGBoost"]     = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    _explainers["XGBoost"] = joblib.load(os.path.join(MODELS_DIR, "xgboost_explainer.pkl"))

    # LightGBM
    _models["LightGBM"]     = joblib.load(os.path.join(MODELS_DIR, "lightgbm_model.pkl"))
    _explainers["LightGBM"] = joblib.load(os.path.join(MODELS_DIR, "lightgbm_explainer.pkl"))

    # Autoencoder
    _models["Autoencoder"] = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "autoencoder_model.keras")
    )
    with open(os.path.join(MODELS_DIR, "autoencoder_threshold.json")) as f:
        _meta["autoencoder_threshold"] = json.load(f)["threshold"]

    # Metrics summaries
    for name in ["xgboost", "lightgbm", "autoencoder"]:
        with open(os.path.join(MODELS_DIR, f"{name}_metrics.json")) as f:
            _meta[name] = json.load(f)

    print("  ✅ All models loaded")


def get_model(name: str):
    if not _models:
        load_all_models()
    return _models.get(name)


def get_explainer(name: str):
    return _explainers.get(name)


def get_scalers():
    return _scalers


def get_meta(name: str):
    return _meta.get(name.lower(), {})


def get_all_metrics() -> dict:
    if not _meta:
        load_all_models()
    return {
        "XGBoost"    : _meta.get("xgboost",     {}),
        "LightGBM"   : _meta.get("lightgbm",    {}),
        "Autoencoder": _meta.get("autoencoder",  {}),
    }


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_input(request_data: dict) -> np.ndarray:
    """
    Takes raw user input from the API request,
    scales Amount and Time, returns a numpy array ready for prediction.
    """
    scalers = get_scalers()

    amount_scaled = scalers["amount"].transform([[request_data["amount"]]])[0][0]
    time_scaled   = scalers["time"].transform([[request_data["time"]]])[0][0]

    features = [time_scaled, request_data["v1"], request_data["v2"],
                request_data["v3"],  request_data["v4"],  request_data["v5"],
                request_data["v6"],  request_data["v7"],  request_data["v8"],
                request_data["v9"],  request_data["v10"], request_data["v11"],
                request_data["v12"], request_data["v13"], request_data["v14"],
                request_data["v15"], request_data["v16"], request_data["v17"],
                request_data["v18"], request_data["v19"], request_data["v20"],
                request_data["v21"], request_data["v22"], request_data["v23"],
                request_data["v24"], request_data["v25"], request_data["v26"],
                request_data["v27"], request_data["v28"], amount_scaled]

    return np.array(features).reshape(1, -1)


# ── Predict ────────────────────────────────────────────────────────────────────
def predict(model_name: str, X: np.ndarray) -> Tuple[str, float, float]:
    """
    Runs inference for the specified model.
    Returns: (prediction label, risk_score, confidence_score)
    """
    model = get_model(model_name)
    if model is None:
        load_all_models()
        model = get_model(model_name)

    if model_name in ("XGBoost", "LightGBM"):
        prob        = float(model.predict_proba(X)[0][1])
        prediction  = "FRAUD" if prob >= 0.5 else "NORMAL"
        risk_score  = prob
        confidence  = prob if prediction == "FRAUD" else 1 - prob

    elif model_name == "Autoencoder":
        recon       = model.predict(X, verbose=0)
        # MUST match the combined metric used during training threshold selection
        # (see train_autoencoder.py → compute_reconstruction_errors)
        mse_arr     = np.mean(np.power(X - recon, 2), axis=1)
        mae_arr     = np.mean(np.abs(X - recon), axis=1)
        max_err_arr = np.max(np.abs(X - recon), axis=1)
        combined    = float((0.5 * mse_arr + 0.3 * mae_arr + 0.2 * max_err_arr)[0])

        threshold   = _meta.get("autoencoder_threshold", 0.5)
        prediction  = "FRAUD" if combined > threshold else "NORMAL"
        # Normalize combined error to 0-1 range for risk score
        risk_score  = min(combined / (threshold * 3), 1.0)
        confidence  = risk_score if prediction == "FRAUD" else 1 - risk_score

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return prediction, round(risk_score, 4), round(confidence, 4)
