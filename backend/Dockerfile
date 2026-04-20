# ══════════════════════════════════════════════════════════════
#  FraudX Analyst - Backend Dockerfile
#  Deploys FastAPI + ML models to Render (free tier)
# ══════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY ml/ ml/
COPY knowledge_base/ knowledge_base/
COPY data/ data/

# Create __init__.py files if missing
RUN touch app/__init__.py app/api/__init__.py app/services/__init__.py

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
