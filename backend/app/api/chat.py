"""
FraudX Analyst - Chat API
===========================
POST /api/v1/chat

Multi-agent RAG chatbot that:
1. Classifies the question type (general fraud knowledge vs simulation-specific)
2. Retrieves relevant context from Pinecone knowledge base
3. Generates a response using Gemini with the retrieved context
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import os, json
from dotenv import load_dotenv

from app.models import ChatRequest, ChatResponse, SimulationHistory
from app.database import get_db, ensure_device
from app.services.gemini_service import chat_response

load_dotenv()

router = APIRouter()

# ── Pinecone setup (lazy loaded) ───────────────────────────────────────────────
_pinecone_index = None

def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
        index_name = os.getenv("PINECONE_INDEX_NAME", "fraudx-knowledge")
        _pinecone_index = pc.Index(index_name)
        return _pinecone_index
    except Exception as e:
        print(f"  ⚠️ Pinecone not available: {e}")
        return None


async def retrieve_context(query: str, top_k: int = 3) -> list[str]:
    """
    Searches Pinecone for relevant knowledge base chunks.
    Falls back to empty context if Pinecone is not configured.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

        from app.services.gemini_service import get_embedding
        embedding = get_embedding(query)
        if not embedding:
            return []

        index = get_pinecone_index()
        if index is None:
            return []

        results = index.query(
            vector          = embedding,
            top_k           = top_k,
            include_metadata= True,
        )

        return [match["metadata"].get("text", "") for match in results["matches"]]

    except Exception as e:
        print(f"  ⚠️ Context retrieval failed: {e}")
        return []


# ── Fallback context — used when Pinecone is not yet set up ───────────────────
FALLBACK_CONTEXT = [
    """Credit card fraud detection uses machine learning models to identify suspicious transactions.
XGBoost and LightGBM are supervised models trained on labelled fraud/normal data.
An Autoencoder is an unsupervised model that learns normal behaviour and flags anomalies.""",

    """SHAP (SHapley Additive exPlanations) explains which features most influenced a model's prediction.
Positive SHAP values push the prediction toward fraud; negative values push toward normal.
Features V1-V28 are PCA-transformed variables from the original transaction data.""",

    """Common credit card fraud types include card-not-present fraud, account takeover, 
identity theft, skimming, and phishing. If you suspect fraud, contact your bank immediately,
freeze your card, and review recent transactions.""",
]


# ── POST /chat ─────────────────────────────────────────────────────────────────
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Main chatbot endpoint called by the Flutter Chat screen.

    Agents:
    1. Router agent   → decides if question is about a simulation or general knowledge
    2. Retrieval agent → fetches relevant context from Pinecone
    3. Answer agent   → Gemini generates final answer with context
    """

    # ── Auto-register device ─────────────────────────────────────────────────
    await ensure_device(db, request.device_id)

    # ── Agent 1: Get simulation context if simulation_id provided ─────────────
    simulation_context = None
    if request.simulation_id:
        try:
            result = await db.execute(
                select(SimulationHistory)
                .where(SimulationHistory.simulation_id == request.simulation_id)
            )
            sim = result.scalar_one_or_none()
            if sim:
                top_features = json.loads(sim.top_features) if sim.top_features else []
                simulation_context = (
                    f"Simulation ID: {sim.simulation_id}\n"
                    f"Prediction: {sim.prediction_result}\n"
                    f"Risk Score: {sim.risk_score:.1%}\n"
                    f"Amount: ${sim.transaction_amount:.2f}\n"
                    f"AI Explanation: {sim.xai_explanation}\n"
                    f"Top Features: {json.dumps(top_features[:3])}"
                )
        except Exception as e:
            print(f"  ⚠️ Could not load simulation context: {e}")

    # ── Agent 2: Retrieve context from Pinecone ───────────────────────────────
    context_docs = await retrieve_context(request.message)

    # Fall back to hardcoded context if Pinecone returns nothing
    if not context_docs:
        context_docs = FALLBACK_CONTEXT

    # ── Agent 3: Generate response with Gemini ────────────────────────────────
    reply = await chat_response(
        user_message       = request.message,
        context_docs       = context_docs,
        simulation_context = simulation_context,
        chat_history       = request.chat_history,
    )

    return ChatResponse(
        reply   = reply,
        sources = ["FraudX Knowledge Base"] if context_docs else [],
    )


# ── GET /chat/suggestions ──────────────────────────────────────────────────────
@router.get("/chat/suggestions")
async def get_suggestions():
    """
    Returns suggested questions for the Flutter Chat screen quick-reply buttons.
    """
    return {
        "suggestions": [
            "What is credit card fraud?",
            "How does XGBoost detect fraud?",
            "What does SHAP mean?",
            "Why was my transaction flagged as fraud?",
            "What should I do if my card is compromised?",
            "How accurate is the Autoencoder model?",
            "What are the most common types of card fraud?",
            "How is the risk score calculated?",
        ]
    }
