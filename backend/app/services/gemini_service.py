"""
FraudX Analyst - Gemini Service
================================
Uses Google Gemini 1.5 Flash to generate:
  1. Natural language explanations of fraud prediction results
  2. Answers to fraud-related questions in the RAG chatbot
"""

import os
from typing import List, Optional
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── Configure Gemini client ────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "models/gemini-2.5-flash"

# ── Prediction explanation ─────────────────────────────────────────────────────
async def explain_prediction(
    prediction      : str,
    risk_score      : float,
    confidence_score: float,
    model_name      : str,
    top_features    : List[dict],
    amount          : float,
    location        : Optional[str] = None,
) -> str:
    """
    Takes the raw ML prediction output and generates a clear,
    human-readable explanation for the Flutter app user.
    """
    # Map V-features to likely real-world meanings for better explanations
    feature_meanings = {
        'V1': 'Transaction frequency pattern', 'V2': 'Merchant risk profile',
        'V3': 'Geographic consistency', 'V4': 'Spending amount deviation',
        'V5': 'Card usage pattern', 'V6': 'Transaction velocity',
        'V7': 'Time-of-day risk factor', 'V8': 'Merchant category risk',
        'V9': 'Account behaviour anomaly', 'V10': 'Cardholder profile deviation',
        'V11': 'Cross-border indicator', 'V12': 'Spending category anomaly',
        'V13': 'Transaction sequence pattern', 'V14': 'Historical fraud correlation',
        'V15': 'Payment channel risk', 'V16': 'Device fingerprint anomaly',
        'V17': 'Behavioural biometric deviation', 'V18': 'Session risk factor',
        'V19': 'Authentication pattern', 'V20': 'Recurring payment indicator',
        'V21': 'IP geolocation risk', 'V22': 'Time since last transaction',
        'V23': 'Merchant trust score', 'V24': 'Card-not-present indicator',
        'V25': 'Account age factor', 'V26': 'Chargeback history',
        'V27': 'Transaction rounding pattern', 'V28': 'Velocity check score',
    }

    features_text = "\n".join([
        f"  - {f['feature']} ({feature_meanings.get(f['feature'], 'Unknown factor')}): "
        f"impact={f['impact']:.4f} ({'pushes toward FRAUD (red/positive)' if f['impact'] > 0 else 'pushes toward NORMAL (green/negative)'}), "
        f"raw value={f['value']:.4f}"
        for f in top_features[:5]
    ])

    prompt = f"""You are FraudX, an expert AI fraud detection analyst.
    Provide a detailed, specific analysis of this credit card transaction. Do NOT be generic.

    Transaction Details:
    - Amount: ${amount:.2f}
    - Model Used: {model_name}
    - Prediction: {prediction}
    - Risk Score: {risk_score:.1%}
    - Confidence: {confidence_score:.1%}

    Key Risk Factors from SHAP Analysis (these are the features that most influenced the prediction):
    {features_text}

    IMPORTANT — Your explanation MUST include ALL of the following:

    1. State clearly whether this transaction is FRAUD or NORMAL with the risk percentage

    2. Explain what SHAP Feature Importance means in simple terms:
    - "The chart above shows which transaction characteristics most influenced the prediction"
    - Red bars (positive values) = factors pushing toward FRAUD
    - Green bars (negative values) = factors pushing toward NORMAL
    - Longer bars = stronger influence on the prediction

    3. For EACH of the top 3 features shown, explain:
    - What the feature name means (use the descriptive name, e.g. V14 means "Historical fraud correlation")
    - What its value indicates about this specific transaction
    - Whether it pushed toward fraud or normal and why that matters

    4. Summarise the overall pattern: why the combination of these factors led to the final prediction

    5. If FRAUD: advise the user to contact their bank and freeze the card
    If NORMAL: reassure the user but remind them to monitor statements

    Do NOT use markdown bold (no ** markers). Write in plain text paragraphs.
    Vary your phrasing each time — do not use the same template.
    Reference the actual SHAP values and feature names from the data above.
    Keep the tone professional but accessible to non-technical users."""

    try:
        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"  ⚠️ Gemini explanation failed: {e}")
        verdict = "potentially fraudulent" if prediction == "FRAUD" else "normal"
        return (
            f"This transaction has been classified as {verdict} with a risk score of "
            f"{risk_score:.1%} by the {model_name} model. "
            f"{'Please review this transaction carefully and contact your bank if you did not authorise it.' if prediction == 'FRAUD' else 'No immediate action is required.'}"
        )


# ── Chat response ──────────────────────────────────────────────────────────────
async def chat_response(
    user_message      : str,
    context_docs      : List[str],
    simulation_context: Optional[str] = None,
    chat_history      : Optional[List[dict]] = None,
) -> str:
    """
    Generates a RAG chatbot response using retrieved knowledge base docs.
    Includes conversation history for multi-turn context.
    """
    context_text = "\n\n".join(context_docs) if context_docs else "No specific context available."

    simulation_text = ""
    if simulation_context:
        simulation_text = f"\nRecent Simulation Result:\n{simulation_context}\n"

    # Build conversation history string (last 6 messages for context window management)
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]  # Keep last 6 messages to avoid token limits
        history_lines = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "FraudX"
            history_lines.append(f"{role}: {msg.get('content', '')}")
        history_text = f"\nConversation History:\n" + "\n".join(history_lines) + "\n"

    prompt = f"""You are FraudX Assistant, an expert in credit card fraud detection and cybersecurity.
Answer the user's question using the provided context. Be helpful, accurate, and concise.

Knowledge Base Context:
{context_text}
{simulation_text}
{history_text}
User Question: {user_message}

Guidelines:
- Answer based on the context provided
- If the question is about a specific simulation result, refer to it directly
- If the user refers to something from previous messages, use the conversation history to understand the context
- Keep answers clear and practical
- If you don't know something, say so honestly
- Use bullet points for lists, keep paragraphs short"""

    try:
        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"  ⚠️ Gemini chat failed: {e}")
        return "I'm sorry, I couldn't process your question at the moment. Please try again."


# ── Embedding for Pinecone RAG ─────────────────────────────────────────────────
def get_embedding(text: str) -> List[float]:
    """
    Generates a text embedding using Google's gemini-embedding-001 model.
    MUST match the dimension used in upload_knowledge.py (2048).
    """
    try:
        from google.genai import types
        response = client.models.embed_content(
            model    = "models/gemini-embedding-001",
            contents = text,
            config   = types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=2048,
            ),
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"  ⚠️ Embedding failed: {e}")
        return []
