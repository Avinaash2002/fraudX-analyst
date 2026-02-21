"""
FraudX Analyst - Predict API
==============================
POST /api/v1/predict
Runs fraud detection and returns prediction + SHAP + Gemini explanation.
"""

import json, time, uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import PredictRequest, PredictResponse, SimulationHistory, MLModel
from app.database import get_db, ensure_device
from app.services.ml_service import preprocess_input, predict, load_all_models, get_model
from app.services.xai_service import get_top_features, get_shap_explanation
from app.services.gemini_service import explain_prediction

router = APIRouter()

# Load models on first import
_models_loaded = False


def ensure_models():
    global _models_loaded
    if not _models_loaded:
        load_all_models()
        _models_loaded = True


# ── POST /predict ──────────────────────────────────────────────────────────────
@router.post("/predict", response_model=PredictResponse)
async def predict_fraud(request: PredictRequest, db: AsyncSession = Depends(get_db)):
    """
    Main prediction endpoint called by the Flutter Simulate screen.
    
    1. Preprocesses input (scales Amount and Time)
    2. Runs selected ML model
    3. Generates SHAP feature importance
    4. Gets Gemini AI explanation
    5. Saves result to Supabase
    6. Returns full result to Flutter
    """
    ensure_models()
    t_start = time.time()

    # 1. Validate model name
    valid_models = ["XGBoost", "LightGBM", "Autoencoder"]
    if request.model_name not in valid_models:
        raise HTTPException(status_code=400, detail=f"Model must be one of {valid_models}")

    # 2. Preprocess
    X = preprocess_input(request.dict())

    # 3. Predict
    prediction, risk_score, confidence = predict(request.model_name, X)

    # 4. SHAP top features
    top_features  = get_top_features(request.model_name, X, top_n=5)
    shap_features = get_shap_explanation(request.model_name, X, top_n=10)

    # 5. Gemini explanation
    ai_explanation = await explain_prediction(
        prediction       = prediction,
        risk_score       = risk_score,
        confidence_score = confidence,
        model_name       = request.model_name,
        top_features     = top_features,
        amount           = request.amount,
        location         = request.location,
    )

    processing_time = round(time.time() - t_start, 3)
    simulation_id   = str(uuid.uuid4())

    # 6. Look up model_id from database
    model_id = None
    try:
        result   = await db.execute(
            select(MLModel).where(MLModel.model_name == request.model_name)
        )
        ml_model = result.scalar_one_or_none()
        if ml_model:
            model_id = ml_model.model_id
    except Exception:
        pass    # model_id stays None if not found — non-critical

    # 7. Auto-register device + save to Supabase
    try:
        await ensure_device(db, request.device_id)
        record = SimulationHistory(
            simulation_id      = simulation_id,
            device_id          = request.device_id,
            model_id           = model_id,
            transaction_amount = request.amount,
            location           = request.location,
            prediction_result  = prediction,
            risk_score         = risk_score,
            xai_explanation    = ai_explanation,
            confidence_score   = confidence,
            processing_time    = processing_time,
            transaction_time   = request.time,
            transaction_data   = json.dumps(request.dict()),
            card_number        = request.card_number,
            top_features       = json.dumps(top_features),
        )
        db.add(record)
        await db.commit()
    except Exception as e:
        print(f"  ⚠️ Could not save to DB: {e}")
        await db.rollback()

    return PredictResponse(
        simulation_id    = simulation_id,
        prediction       = prediction,
        risk_score       = risk_score,
        confidence_score = confidence,
        processing_time  = processing_time,
        top_features     = top_features,
        shap_values      = shap_features,
        ai_explanation   = ai_explanation,
    )
