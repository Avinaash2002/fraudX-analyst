"""
FraudX Analyst - Database Models & Pydantic Schemas
=====================================================
SQLAlchemy models → define database tables
Pydantic schemas  → define API request/response shapes
"""

import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel

from app.database import Base


# ══════════════════════════════════════════════════════════════════════════════
#  SQLAlchemy Models (Database Tables)
# ══════════════════════════════════════════════════════════════════════════════

class User(Base):
    __tablename__ = "users"

    device_id      = Column(String(25), primary_key=True)
    last_active    = Column(DateTime, default=datetime.utcnow)
    app_version    = Column(String(5), default="1.0.0")
    account_amount = Column(Float, default=0)

class Dataset(Base):
    __tablename__ = "datasets"

    dataset_id      = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_name    = Column(String(255), nullable=False)
    record_count    = Column(Integer)
    is_imbalanced   = Column(Boolean, default=True)
    source_url      = Column(String(500))
    fraud_count     = Column(Integer)
    normal_count    = Column(Integer)
    imbalance_ratio = Column(Float)
    storage_path    = Column(String(500))
    file_size       = Column(Float)
    upload_date     = Column(DateTime, default=datetime.utcnow)
    feature_count   = Column(Integer)


class MLModel(Base):
    __tablename__ = "ml_models"

    model_id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name           = Column(String(100), nullable=False)
    accuracy             = Column(Float)
    precision            = Column(Float)
    recall               = Column(Float)
    f1_score             = Column(Float)
    auc                  = Column(Float)
    training_date        = Column(DateTime, default=datetime.utcnow)
    algorithm_type       = Column(String(50))   # supervised / unsupervised
    dataset_id           = Column(UUID(as_uuid=True), ForeignKey("datasets.dataset_id"), nullable=True)
    description          = Column(Text)
    training_duration    = Column(Float)
    mlflow_run_id        = Column(String(255))
    mlflow_experiment_id = Column(String(255))
    mlflow_model_uri     = Column(String(500))


class SimulationHistory(Base):
    __tablename__ = "simulation_history"

    simulation_id     = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id         = Column(String(25), ForeignKey("users.device_id"))
    model_id          = Column(UUID(as_uuid=True), ForeignKey("ml_models.model_id"), nullable=True)
    timestamp         = Column(DateTime, default=datetime.utcnow)
    transaction_amount= Column(Float)
    location          = Column(String(255))
    prediction_result = Column(String(10))      # FRAUD or NORMAL
    risk_score        = Column(Float)
    xai_explanation   = Column(Text)
    confidence_score  = Column(Float)
    processing_time   = Column(Float)
    transaction_time  = Column(Float)
    transaction_data  = Column(Text)            # JSON string of all features
    card_number       = Column(String(20))
    top_features      = Column(Text)            # JSON string of top SHAP features


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    content_id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title              = Column(String(500), nullable=False)
    category           = Column(String(100))
    upload_date        = Column(DateTime, default=datetime.utcnow)
    content_type       = Column(String(50))
    format             = Column(String(20))
    storage_path       = Column(String(500))
    raw_content        = Column(Text)
    is_embedded        = Column(Boolean, default=False)
    pinecone_index_name= Column(String(255))
    embedding_status   = Column(String(50), default="pending")
    chunk_count        = Column(Integer, default=0)
    publish_date       = Column(DateTime)
    last_updated       = Column(DateTime, default=datetime.utcnow)
    tags               = Column(String(500))


# ══════════════════════════════════════════════════════════════════════════════
#  Pydantic Schemas (API request / response shapes)
# ══════════════════════════════════════════════════════════════════════════════

# ── Prediction ─────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    model_name    : str               # "XGBoost" | "LightGBM" | "Autoencoder"
    amount        : float
    time          : float
    v1            : float
    v2            : float
    v3            : float
    v4            : float
    v5            : float
    v6            : float
    v7            : float
    v8            : float
    v9            : float
    v10           : float
    v11           : float
    v12           : float
    v13           : float
    v14           : float
    v15           : float
    v16           : float
    v17           : float
    v18           : float
    v19           : float
    v20           : float
    v21           : float
    v22           : float
    v23           : float
    v24           : float
    v25           : float
    v26           : float
    v27           : float
    v28           : float
    device_id     : Optional[str] = None
    card_number   : Optional[str] = None
    location      : Optional[str] = None


class PredictResponse(BaseModel):
    simulation_id     : str
    prediction        : str           # "FRAUD" or "NORMAL"
    risk_score        : float         # 0.0 – 1.0
    confidence_score  : float
    processing_time   : float
    top_features      : List[dict]    # [{"feature": "V14", "value": -3.2, "impact": 0.45}]
    shap_values       : Optional[List[dict]] = None
    lime_values       : Optional[List[dict]] = None
    ai_explanation    : str           # Gemini natural language explanation


# ── Models ─────────────────────────────────────────────────────────────────────
class ModelMetricsResponse(BaseModel):
    model_id        : Optional[str]
    model_name      : str
    accuracy        : float
    precision       : float
    recall          : float
    f1_score        : float
    auc             : float
    algorithm_type  : str
    training_date   : Optional[str]
    description     : Optional[str]


# ── History ────────────────────────────────────────────────────────────────────
class HistoryResponse(BaseModel):
    simulation_id     : str
    timestamp         : str
    model_name        : Optional[str]
    transaction_amount: float
    prediction_result : str
    risk_score        : float
    location          : Optional[str]
    card_number       : Optional[str]


# ── Chat ───────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message           : str
    device_id         : Optional[str] = None
    simulation_id     : Optional[str] = None   # attach a simulation result to the query
    chat_history      : Optional[List[dict]] = None  # [{"role": "user"|"assistant", "content": "..."}]


class ChatResponse(BaseModel):
    reply             : str
    sources           : Optional[List[str]] = None
