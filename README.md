# 🛡️ FraudX Analyst

**AI-Powered Credit Card Fraud Detection Mobile Application**

An end-to-end fraud detection system combining machine learning, SHAP explainability, and a RAG-powered AI chatbot — built as a Final Year Project at Universiti Malaysia Sarawak (UNIMAS).

---

## 📌 Overview

FraudX Analyst is a Flutter mobile application that detects, analyses, and explains potentially fraudulent credit card transactions using three machine learning models. The system goes beyond simple prediction by providing visual feature importance through SHAP, human-readable AI explanations via Google Gemini, and a conversational chatbot grounded in a curated fraud detection knowledge base.

**Key Highlights:**
- 3 ML Models: XGBoost, LightGBM, Autoencoder
- SHAP Feature Attribution for every prediction
- RAG Chatbot (Gemini + Pinecone) for contextual Q&A
- Interactive 16-step onboarding tutorial
- Smart model upgrade mechanism
- Deployed on Render (Docker) with Supabase persistence

---

## 🏗️ System Architecture

```
Flutter App (Android)
       │
       ▼ HTTPS
┌──────────────────────────────────────┐
│  FastAPI Backend (Docker on Render)  |
│                                      │
│  ┌───────────┐  ┌──────────────────┐ │
│  │ ML Models │  │ SHAP Explainer   │ │
│  │ XGBoost   │  │ TreeExplainer    │ │
│  │ LightGBM  │  │ KernelExplainer  │ │
│  │Autoencoder│  │                  │ │
│  └─────┬─────┘  └────────┬─────────┘ │
│        │                 │           │
│        ▼                 ▼           │
│  ┌──────────────────────────────┐    │
│  │ Google Gemini API            │    │
│  │ • AI Explanations            │    │
│  │ • RAG Chat Responses         │    │
│  │ • Embedding Generation       │    │
│  └──────────────────────────────┘    │
└──────┬──────────────┬────────────────┘
       │              │
       ▼              ▼
   Supabase       Pinecone
  (PostgreSQL)   (Vector DB)
```

---

## ✨ Features

### Home Dashboard
- Real-time statistics: safe/fraud counts, model accuracy, AUC-ROC
- Clickable recent transactions with full detail bottom sheet
- Pull-to-refresh and auto-reload on tab switch
- Retry mechanism for Render cold start

### Transaction Simulation
- Select from 3 ML models or auto-select best (highest F1)
- Load real unseen transactions from held-out test set
- Real-time input validation (amount, time, card number)
- SHAP feature importance chart (red = fraud, green = normal)
- AI explanation with feature meanings (e.g., V14 = Historical fraud correlation)

### Model Training
- Train on built-in Kaggle dataset (284,807 transactions) or custom CSV
- Optuna hyperparameter tuning (10 trials per model)
- Smart upgrade: only overwrites model if new F1 > existing F1
- Dataset format guide with V1-V28 PCA descriptions
- PDF training report export

### Model Comparison
- Side-by-side bar charts for 5 metrics
- Information dialogs explaining each metric
- Best model highlighted with trophy badge
- PDF comparison report export

### RAG AI Chatbot
- Powered by Google Gemini + Pinecone vector database
- Curated fraud detection knowledge base
- Maintains conversation context (last 6 messages)
- Explains specific simulation results when asked

### Interactive Tutorial
- 16-step guided onboarding for first-time users
- Hands-on interaction: user performs real simulation during guide
- Blinking indicators, floating Continue button, navigation restrictions
- Shows only once per device (SharedPreferences)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Flutter 3.29, Dart, Provider |
| Backend | Python 3.11, FastAPI |
| ML Models | XGBoost, LightGBM, TensorFlow (Autoencoder) |
| Explainability | SHAP (TreeExplainer, KernelExplainer) |
| Experiment Tracking | MLflow (local training phase) |
| Generative AI | Google Gemini 2.5 Flash |
| Vector Database | Pinecone (2048-dim embeddings) |
| Database | Supabase (PostgreSQL) |
| Deployment | Docker, Render |
| Hyperparameter Tuning | Optuna |

---

## 📁 Project Structure

```
fraudX-analyst/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── database.py          # Supabase async connection
│   │   ├── models.py            # Pydantic models
│   │   ├── api/
│   │   │   ├── predict.py       # POST /predict endpoint
│   │   │   ├── chat.py          # POST /chat endpoint
│   │   │   ├── history.py       # GET /history endpoint
│   │   │   └── train.py         # GET /models, POST /train
│   │   └── services/
│   │       ├── ml_service.py    # Model loading & inference
│   │       ├── xai_service.py   # SHAP explanations
│   │       └── gemini_service.py # Gemini AI integration
│   ├── ml/
│   │   ├── training/
│   │   │   ├── preprocess.py    # Data preprocessing (no leakage)
│   │   │   ├── train_all.py     # Full training pipeline
│   │   │   ├── train_xgboost.py # XGBoost + Optuna
│   │   │   ├── train_lightgbm.py # LightGBM + Optuna
│   │   │   ├── train_autoencoder.py # Autoencoder
│   │   │   └── models_saved/    # Serialised models & metrics
│   │   └── data/
│   │       └── test_sample.csv  # 574-row held-out sample
│   ├── knowledge_base/
│   │   ├── knowledge_content.py # Fraud detection knowledge
│   │   └── upload_knowledge.py  # Pinecone upload script
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
│
└── fraudx_analyst_app/
    ├── lib/
    │   ├── main.dart
    │   ├── config/
    │   │   └── api_config.dart   # API URL configuration
    │   ├── models/
    │   │   └── models.dart       # Data models
    │   ├── providers/
    │   │   └── app_provider.dart  # State management
    │   ├── services/
    │   │   ├── api_service.dart   # HTTP client
    │   │   ├── pdf_report_service.dart
    │   │   └── tutorial_service.dart
    │   ├── screens/
    │   │   ├── start_screen.dart
    │   │   ├── home_screen.dart
    │   │   ├── simulate_screen.dart
    │   │   ├── train_screen.dart
    │   │   ├── models_screen.dart
    │   │   ├── chat_screen.dart
    │   │   ├── history_screen.dart
    │   │   └── user_guide_screen.dart
    │   └── widgets/
    │       ├── animated_bot.dart
    │       └── tutorial_overlay.dart
    ├── assets/
    │   └── app_icon.png
    └── pubspec.yaml
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Flutter 3.29+
- Git

### Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Create a `.env` file from the example:
```bash
cp .env.example .env
# Fill in your API keys
```

Upload the knowledge base:
```bash
python knowledge_base/upload_knowledge.py
```

Start the server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Flutter App Setup

```bash
cd fraudx_analyst_app
flutter pub get
```

Update `lib/config/api_config.dart` with your backend URL, then:

```bash
flutter run                    # Run on connected device
# OR
flutter build apk --release   # Build release APK
```

---

## 📊 Model Performance

Trained on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud).

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 99.95% | 87.14% | 82.43% | 84.72% | 0.9787 |
| LightGBM | 99.95% | 88.24% | 81.08% | 84.51% | 0.9722 |
| Autoencoder | 99.81% | 44.30% | 47.30% | 45.75% | 0.9490 |

---

## 🌐 Deployment

The backend is deployed on [Render](https://render.com) using Docker:

- **Live URL:** https://fraudx-analyst.onrender.com
- **API Docs:** https://fraudx-analyst.onrender.com/docs

> ⚠️ Render free tier sleeps after 15 minutes of inactivity. First request after sleep takes ~30 seconds.

---

## 📱 Download

Download the latest release APK from the [Releases](https://github.com/Avinaash2002/fraudX-analyst/releases) page.

**Requirements:** Android 8.0+ with internet connection.

---

## 📝 Dataset

The application uses the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
- 284,807 transactions by European cardholders (September 2013)
- 492 fraud cases (0.17%)
- 28 PCA-transformed features (V1-V28) + Time + Amount
- Binary classification: Class 0 (Normal) / Class 1 (Fraud)

The full dataset is not included in this repository due to size. Download it from Kaggle and place it at `backend/ml/data/creditcard.csv` for local training.

---

## 👤 Author

**Avinaash A/L Loganathan**
- Faculty of Computer Science & Information Technology
- Universiti Malaysia Sarawak (UNIMAS)
- Email: avinaash.loganathan24@gmail.com

## 👤 Supervisor

**Prof Dr Jane Labadin**
- Faculty of Computer Science & Information Technology
- Universiti Malaysia Sarawak (UNIMAS)
- Email: ljane@unimas.my

---

## 📄 License

This project was developed as a Final Year Project (FYP) for academic purposes at UNIMAS. All rights reserved.
