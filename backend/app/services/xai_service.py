"""
FraudX Analyst - XAI Service
==============================
Generates SHAP and LIME explanations for any model prediction.
Returns top features with their impact values for display in the Flutter app.
"""

import numpy as np
from typing import List, Dict
import shap
import lime
import lime.lime_tabular

from app.services.ml_service import get_explainer, get_scalers, get_model


# ── SHAP ───────────────────────────────────────────────────────────────────────
def get_shap_explanation(model_name: str, X: np.ndarray, top_n: int = 10) -> List[Dict]:
    """
    Uses the pre-built SHAP TreeExplainer to compute feature importance.
    Returns top N features sorted by absolute impact on prediction.

    Works for XGBoost and LightGBM only (tree-based models).
    For Autoencoder we use a simplified feature difference approach.
    """
    feature_names = get_scalers().get("features", [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"])

    if model_name in ("XGBoost", "LightGBM"):
        explainer   = get_explainer(model_name)
        shap_values = explainer.shap_values(X)

        # LightGBM returns list [class0, class1] — we want class 1 (fraud)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        results = []
        for i, (name, value, shap_val) in enumerate(zip(feature_names, X[0], sv)):
            results.append({
                "feature": name,
                "value"  : round(float(value), 4),
                "impact" : round(float(shap_val), 4),
            })

        # Sort by absolute impact, return top N
        results.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return results[:top_n]

    elif model_name == "Autoencoder":
        # For Autoencoder: per-feature squared reconstruction error shows
        # which features the model struggled most to reconstruct (anomalous features).
        # Note: the overall prediction threshold uses a combined metric
        # (0.5*MSE + 0.3*MAE + 0.2*MaxErr), but for feature-level attribution,
        # squared error per feature is the most interpretable breakdown.
        model  = get_model("Autoencoder")
        recon  = model.predict(X, verbose=0)
        errors = np.power(X - recon, 2)[0]

        results = []
        for i, (name, value, err) in enumerate(zip(feature_names, X[0], errors)):
            results.append({
                "feature": name,
                "value"  : round(float(value), 4),
                "impact" : round(float(err), 6),
            })

        results.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return results[:top_n]

    return []


# ── LIME ───────────────────────────────────────────────────────────────────────
def get_lime_explanation(model_name: str, X: np.ndarray,
                         X_train_sample: np.ndarray = None,
                         top_n: int = 10) -> List[Dict]:
    """
    Uses LIME to explain a single prediction by perturbing the input
    and observing how the model output changes.

    Requires a sample of training data to build the background distribution.
    If not provided, uses the input itself (less accurate but still works).
    """
    feature_names = get_scalers().get("features", [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"])

    if X_train_sample is None:
        # Fallback: use the input repeated as background
        X_train_sample = np.repeat(X, 100, axis=0)

    if model_name in ("XGBoost", "LightGBM"):
        model = get_model(model_name)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data   = X_train_sample,
            feature_names   = feature_names,
            class_names     = ["Normal", "Fraud"],
            mode            = "classification",
            random_state    = 42,
        )

        explanation = explainer.explain_instance(
            data_row        = X[0],
            predict_fn      = model.predict_proba,
            num_features    = top_n,
        )

        results = []
        for feat_name, impact in explanation.as_list():
            # LIME returns "feature_name <= value" style strings — extract just the name
            clean_name = feat_name.split(" ")[0].split(">")[0].split("<")[0].strip()
            results.append({
                "feature": clean_name,
                "impact" : round(float(impact), 4),
            })

        return results

    return []


# ── Combined top features ──────────────────────────────────────────────────────
def get_top_features(model_name: str, X: np.ndarray, top_n: int = 5) -> List[Dict]:
    """
    Returns the most important features for display in the Flutter simulation result.
    Uses SHAP for tree models, reconstruction error for Autoencoder.
    """
    return get_shap_explanation(model_name, X, top_n=top_n)
