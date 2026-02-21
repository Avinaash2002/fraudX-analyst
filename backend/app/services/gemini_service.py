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
    features_text = "\n".join([
        f"  - {f['feature']}: value={f['value']:.4f}, impact={f['impact']:.4f}"
        for f in top_features[:5]
    ])

    location_text = f"Location: {location}" if location else "Location: Not specified"

    prompt = f"""You are FraudX, an AI fraud detection assistant.
Explain this credit card transaction analysis result in clear, simple language for a non-technical user.

Transaction Details:
- Amount: ${amount:.2f}
- {location_text}
- Model Used: {model_name}
- Prediction: {prediction}
- Risk Score: {risk_score:.1%}
- Confidence: {confidence_score:.1%} confident this transaction is {prediction}

Top Factors That Influenced This Decision:
{features_text}

Write a 3-4 sentence explanation that:
1. States clearly whether this transaction appears fraudulent or normal
2. Explains the risk level in simple terms (low/medium/high)
3. Mentions 1-2 of the most important factors in plain English
4. Suggests what the user should do if fraud is detected

Keep the tone professional but easy to understand. Do not use technical jargon.
Do not mention the feature names (V1, V14, etc.) directly — describe what they represent conceptually."""

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
    Generates a text embedding using Google's text-embedding-004 model.
    Used to vectorize knowledge base chunks and user queries for Pinecone search.
    """
    try:
        response = client.models.embed_content(
            model    = "models/gemini-embedding-001",
            contents = text,
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"  ⚠️ Embedding failed: {e}")
        return []
