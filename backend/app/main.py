"""
FraudX Analyst - FastAPI Backend
==================================
Main application entry point.
Registers all routers and sets up CORS for Flutter app.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.database import create_tables
from app.api import predict, train, history, chat


# ── Startup / Shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts
    print("🚀 FraudX Analyst API starting up …")
    await create_tables()
    print("✅ Database tables ready")
    yield
    # Runs once when the server shuts down
    print("👋 FraudX Analyst API shutting down …")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "FraudX Analyst API",
    description = "Credit card fraud detection with XAI and RAG chatbot",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS — allows Flutter app to call this API ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # In production, replace with your Flutter app URL
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(train.router,   prefix="/api/v1", tags=["Training"])
app.include_router(history.router, prefix="/api/v1", tags=["History"])
app.include_router(chat.router,    prefix="/api/v1", tags=["Chat"])


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "message": "FraudX Analyst API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
