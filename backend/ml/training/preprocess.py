"""
FraudX Analyst - Data Preprocessing (FIXED - No Leakage)
==========================================================
Proper train/validation/test split with scalers fit ONLY on training data.

Flow:
1. Load dataset
2. Split into Train (70%) / Temp (30%)
3. Fit scalers ONLY on Train
4. Apply scalers to Train/Temp
5. Split Temp into Validation (15%) / Test (15%)

Result: Clean 70/15/15 split with no data leakage.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, '..', 'data',         'creditcard.csv')
MODELS_DIR  = os.path.join(BASE_DIR, '..', 'models_saved')
os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_preprocess():
    """
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    All numpy arrays. Scalers are fit ONLY on train and saved.
    """

    print("=" * 60)
    print("  STEP 1: Loading Dataset")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"  Total   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Normal  : {(df['Class']==0).sum():,}  ({(df['Class']==0).mean()*100:.3f}%)")
    print(f"  Fraud   : {(df['Class']==1).sum():,}  ({(df['Class']==1).mean()*100:.3f}%)")

    # ── Separate features and target ──────────────────────────────────────────
    feature_names = [c for c in df.columns if c != 'Class']
    X = df[feature_names].copy()
    y = df['Class'].values

    # ── CRITICAL: Split BEFORE scaling ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: Train (70%) / Temp (30%) Split")
    print("=" * 60)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    print(f"  Train : {len(X_train):,} samples  (fraud: {y_train.sum()})")
    print(f"  Temp  : {len(X_temp):,} samples  (fraud: {y_temp.sum()})")

    # ── Fit scalers ONLY on training data ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: Scaling (FIT on Train ONLY)")
    print("=" * 60)

    amount_scaler = StandardScaler()
    time_scaler   = StandardScaler()

    # Fit on TRAIN only
    amount_scaler.fit(X_train[['Amount']])
    time_scaler.fit(X_train[['Time']])

    # Transform ALL sets using the same fitted scaler
    X_train.loc[:, 'Amount'] = amount_scaler.transform(X_train[['Amount']])
    X_train.loc[:, 'Time']   = time_scaler.transform(X_train[['Time']])

    X_temp.loc[:, 'Amount']  = amount_scaler.transform(X_temp[['Amount']])
    X_temp.loc[:, 'Time']    = time_scaler.transform(X_temp[['Time']])

    print("  ✅ Scalers fit on train, applied to train/temp")

    # Save scalers for API
    joblib.dump(amount_scaler, os.path.join(MODELS_DIR, 'amount_scaler.pkl'))
    joblib.dump(time_scaler,   os.path.join(MODELS_DIR, 'time_scaler.pkl'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.pkl'))

    # ── Split Temp into Validation and Test ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4: Validation (15%) / Test (15%) Split from Temp")
    print("=" * 60)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,  # 50% of 30% = 15% of total
        random_state=42,
        stratify=y_temp
    )

    print(f"  Val   : {len(X_val):,} samples  (fraud: {y_val.sum()})")
    print(f"  Test  : {len(X_test):,} samples  (fraud: {y_test.sum()})")

    # Convert to numpy
    X_train = X_train.values
    X_val   = X_val.values
    X_test  = X_test.values

    print("\n" + "=" * 60)
    print("  ✅ Preprocessing Complete - No Data Leakage")
    print("=" * 60)
    print(f"  Train : {len(X_train):,} ({len(X_train)/len(X):.1%})")
    print(f"  Val   : {len(X_val):,} ({len(X_val)/len(X):.1%})")
    print(f"  Test  : {len(X_test):,} ({len(X_test)/len(X):.1%})")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def get_normal_only(X_train, y_train):
    """Returns only normal transactions for Autoencoder training."""
    mask = (y_train == 0)
    X_normal = X_train[mask]
    print(f"\n  ✅ Normal-only training set: {X_normal.shape[0]:,} transactions")
    return X_normal


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, features = load_and_preprocess()
    print("\n  ✅ Preprocessing test passed!")
    print(f"  Features: {len(features)}")
