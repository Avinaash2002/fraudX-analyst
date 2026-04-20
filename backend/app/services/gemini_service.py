"""
FraudX Analyst - Gemini Service (Updated)
==========================================
Uses Google Gemini 1.5 Flash (new google-genai SDK) to generate:
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
        'V1': 'transaction frequency pattern', 'V2': 'merchant risk profile',
        'V3': 'geographic consistency', 'V4': 'spending amount deviation',
        'V5': 'card usage pattern', 'V6': 'transaction velocity',
        'V7': 'time-of-day risk factor', 'V8': 'merchant category risk',
        'V9': 'account behavior anomaly', 'V10': 'cardholder profile deviation',
        'V11': 'cross-border indicator', 'V12': 'spending category anomaly',
        'V13': 'transaction sequence pattern', 'V14': 'historical fraud correlation',
        'V15': 'payment channel risk', 'V16': 'device fingerprint anomaly',
        'V17': 'behavioral biometric deviation', 'V18': 'session risk factor',
        'V19': 'authentication pattern', 'V20': 'recurring payment indicator',
        'V21': 'IP geolocation risk', 'V22': 'time since last transaction',
        'V23': 'merchant trust score', 'V24': 'card-not-present indicator',
        'V25': 'account age factor', 'V26': 'chargeback history',
        'V27': 'transaction rounding pattern', 'V28': 'velocity check score',
    }

    features_text = "\n".join([
        f"  - {feature_meanings.get(f['feature'], f['feature'])}: "
        f"impact={f['impact']:.4f} ({'pushes toward FRAUD' if f['impact'] > 0 else 'pushes toward NORMAL'}), "
        f"raw value={f['value']:.4f}"
        for f in top_features[:5]
    ])

    location_text = f"Location: {location}" if location else "Location: Not specified"

    if prediction == "FRAUD":
        analysis_instruction = f"""Write a detailed 5-6 sentence fraud analysis that:
1. State this transaction has been FLAGGED AS FRAUDULENT with a clear warning tone
2. Explain the specific risk level (e.g. "high risk at 87.3%" or "critical risk at 95.2%")
3. Describe the TOP 2-3 specific factors that triggered the fraud alert — explain WHY each factor is suspicious (e.g. "The transaction shows a significant deviation from the cardholder's typical spending behavior, with the behavioral biometric pattern scoring -14.2, far outside the normal range of -2 to +2")
4. Explain what pattern the model detected (e.g. "This combination of unusual merchant risk profile and abnormal transaction velocity is a classic indicator of card-not-present fraud")
5. Give specific actionable advice (contact bank, freeze card, review recent transactions)
6. Mention this analysis was performed by the {model_name} model using SHAP explainability"""
    else:
        analysis_instruction = """Write a detailed 4-5 sentence analysis that:
1. Confirm this transaction appears LEGITIMATE with a reassuring tone
2. State the risk level clearly (e.g. "low risk at 12.4%")
3. Mention 2 specific factors that indicate legitimacy (e.g. "The transaction frequency pattern and spending behavior are consistent with the cardholder's typical usage")
4. Note any slightly elevated factors if present (e.g. "While the merchant category shows a minor deviation, it remains within acceptable bounds")
5. Remind the user to always monitor their statements"""

    prompt = f"""You are FraudX, an expert AI fraud detection analyst.
Provide a detailed, specific analysis of this credit card transaction. Do NOT be generic.

Transaction Details:
- Amount: ${amount:.2f}
- {location_text}
- Model Used: {model_name}
- Prediction: {prediction}
- Risk Score: {risk_score:.1%}
- Confidence: {confidence_score:.1%}

Key Risk Factors (from SHAP Analysis — these are the features that most influenced the prediction):
{features_text}

{analysis_instruction}

IMPORTANT RULES:
- Be SPECIFIC — reference the actual factors and their values from the SHAP analysis above
- Each explanation must be UNIQUE — vary your phrasing and focus different factors each time
- Never use the raw feature names (V1, V14) — use the descriptive names provided
- Include specific numbers (risk percentages, factor values) to sound authoritative
- Do NOT use markdown bold formatting (no ** markers)"""

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
