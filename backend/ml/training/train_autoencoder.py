"""
FraudX Analyst - Autoencoder Training (FIXED - No Leakage)
============================================================
Proper workflow:
1. Train ONLY on normal transactions from TRAIN set
2. Find threshold using VALIDATION set
3. Evaluate ONCE on TEST set
4. Report full metrics
"""

import os, json, time, warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, precision_recall_curve,
                              classification_report, confusion_matrix, average_precision_score)

from preprocess import load_and_preprocess, get_normal_only

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models_saved')
PLOTS_DIR  = os.path.join(BASE_DIR, 'plots')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


def build_autoencoder(input_dim: int) -> Model:
    """
    Encoder : input_dim â†’ 32 â†’ 16 â†’ 8  (compression)
    Decoder : 8 â†’ 16 â†’ 32 â†’ input_dim  (reconstruction)
    """
    inputs = Input(shape=(input_dim,), name='input')
    
    # Encoder
    x = Dense(32, activation='relu', name='enc_32')(inputs)
    x = Dropout(0.2, name='drop_1')(x)
    x = Dense(16, activation='relu', name='enc_16')(x)
    encoded = Dense(8, activation='relu', name='bottleneck')(x)
    
    # Decoder
    x = Dense(16, activation='relu', name='dec_16')(encoded)
    x = Dropout(0.2, name='drop_2')(x)
    x = Dense(32, activation='relu', name='dec_32')(x)
    decoded = Dense(input_dim, activation='linear', name='output')(x)
    
    autoencoder = Model(inputs, decoded, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def compute_reconstruction_errors(model, X):
    """Combined reconstruction error metric."""
    X_recon = model.predict(X, verbose=0)
    
    mse = np.mean(np.power(X - X_recon, 2), axis=1)
    mae = np.mean(np.abs(X - X_recon), axis=1)
    max_err = np.max(np.abs(X - X_recon), axis=1)
    
    # Weighted combination
    combined = 0.5 * mse + 0.3 * mae + 0.2 * max_err
    return combined


def optimize_threshold(y_true, errors):
    """
    Find optimal threshold by testing across error range.
    Selects threshold that maximizes F1 score.
    """
    best_f1 = 0
    best_threshold = 0
    best_metrics = {}
    
    thresholds_to_test = np.linspace(errors.min(), errors.max(), 200)
    
    for thresh in thresholds_to_test:
        y_pred = (errors > thresh).astype(int)
        if y_pred.sum() == 0:
            continue
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_metrics = {'precision': prec, 'recall': rec, 'f1': f1}
    
    return best_threshold, best_f1, best_metrics


def compute_full_metrics(y_true, y_pred, y_prob):
    """Compute comprehensive metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "accuracy"  : round(accuracy_score(y_true, y_pred), 4),
        "precision" : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"    : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score"  : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc"   : round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc"    : round(average_precision_score(y_true, y_prob), 4),
        "true_negatives"  : int(tn),
        "false_positives" : int(fp),
        "false_negatives" : int(fn),
        "true_positives"  : int(tp),
        "support_normal"  : int((y_true == 0).sum()),
        "support_fraud"   : int((y_true == 1).sum()),
    }


def train_autoencoder():
    print("\n" + "ðŸŸ£ " * 25)
    print("  Autoencoder Training (No Leakage - Validation Tuning)")
    print("ðŸŸ£ " * 25)

    # 1. Load data with proper splits
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_preprocess()
    
    # Get normal transactions from TRAIN set only
    X_normal_train = get_normal_only(X_train, y_train)

    mlflow.set_experiment("FraudX-Models-Fixed")

    with mlflow.start_run(run_name="Autoencoder_NoLeakage"):
        
        # 2. Build and train model
        model = build_autoencoder(X_train.shape[1])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
        ]

        print("\n  ðŸŽ¯ Training autoencoder on normal transactions...")
        t0 = time.time()

        history = model.fit(
            X_normal_train, X_normal_train,
            epochs=50,
            batch_size=256,
            # NOTE: This validation_split is for Keras EarlyStopping only — it holds
            # out 10% of the normal-only training data to monitor reconstruction loss.
            # This is separate from our main validation set (X_val) used for threshold tuning.
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0
        )

        training_time = round(time.time() - t0, 2)
        print(f"  âœ… Training complete: {training_time}s ({len(history.history['loss'])} epochs)")

        # 3. Find threshold on VALIDATION set
        print("\n  ðŸ” Optimizing threshold on VALIDATION set...")
        errors_val = compute_reconstruction_errors(model, X_val)
        threshold, val_f1, threshold_metrics = optimize_threshold(y_val, errors_val)
        
        print(f"  âœ… Optimal threshold: {threshold:.6f}")
        print(f"  âœ… Validation F1: {val_f1:.4f}")
        print(f"     Precision: {threshold_metrics['precision']:.4f}")
        print(f"     Recall: {threshold_metrics['recall']:.4f}")

        # 4. Evaluate on TEST set (FIRST AND ONLY TIME)
        print("\n  ðŸ“Š Final Evaluation on TEST Set")
        print("  " + "-" * 56)
        
        errors_test = compute_reconstruction_errors(model, X_test)
        y_pred = (errors_test > threshold).astype(int)
        
        # Normalize errors for probability scores
        errors_min = errors_test.min()
        errors_max = errors_test.max()
        y_prob = (errors_test - errors_min) / (errors_max - errors_min + 1e-10)
        
        metrics = compute_full_metrics(y_test, y_pred, y_prob)

        # Print results
        print(f"\n  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1 Score  : {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
        print(f"  PR-AUC    : {metrics['pr_auc']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {metrics['true_negatives']:,}  |  FP: {metrics['false_positives']:,}")
        print(f"    FN: {metrics['false_negatives']:,}  |  TP: {metrics['true_positives']:,}")
        print(f"\n  Support:")
        print(f"    Normal: {metrics['support_normal']:,}")
        print(f"    Fraud : {metrics['support_fraud']:,}")
        
        print("\n" + classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

        # 5. Plot metrics
        fraud_errors = errors_test[y_test == 1]
        normal_errors = errors_test[y_test == 0]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Error distribution
        axes[0,0].hist(normal_errors, bins=80, alpha=0.6, label='Normal', color='steelblue', density=True)
        axes[0,0].hist(fraud_errors, bins=80, alpha=0.6, label='Fraud', color='tomato', density=True)
        axes[0,0].axvline(threshold, color='black', linestyle='--', linewidth=2,
                         label=f'Threshold = {threshold:.4f}')
        axes[0,0].set_xlabel('Reconstruction Error', fontsize=12)
        axes[0,0].set_ylabel('Density', fontsize=12)
        axes[0,0].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[0,0].legend(fontsize=11)
        axes[0,0].set_yscale('log')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[0,1].plot(fpr, tpr, linewidth=2, label=f'AUC = {metrics["auc_roc"]:.4f}')
        axes[0,1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0,1].set_xlabel('False Positive Rate', fontsize=12)
        axes[0,1].set_ylabel('True Positive Rate', fontsize=12)
        axes[0,1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0,1].legend(fontsize=11)
        axes[0,1].grid(alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        axes[1,0].plot(recall, precision, linewidth=2, color='green',
                      label=f'PR-AUC = {metrics["pr_auc"]:.4f}')
        axes[1,0].set_xlabel('Recall', fontsize=12)
        axes[1,0].set_ylabel('Precision', fontsize=12)
        axes[1,0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1,0].legend(fontsize=11)
        axes[1,0].grid(alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        im = axes[1,1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=axes[1,1])
        classes = ['Normal', 'Fraud']
        tick_marks = np.arange(len(classes))
        axes[1,1].set_xticks(tick_marks)
        axes[1,1].set_xticklabels(classes)
        axes[1,1].set_yticks(tick_marks)
        axes[1,1].set_yticklabels(classes)
        axes[1,1].set_ylabel('True Label', fontsize=12)
        axes[1,1].set_xlabel('Predicted Label', fontsize=12)
        axes[1,1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        thresh_cm = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            axes[1,1].text(j, i, format(cm[i, j], 'd'),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh_cm else "black",
                          fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        eval_plot = os.path.join(PLOTS_DIR, 'autoencoder_evaluation.png')
        plt.savefig(eval_plot, bbox_inches='tight', dpi=150)
        plt.close()

        # 6. Training loss plot
        plt.figure(figsize=(10, 4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Autoencoder Training Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        loss_plot = os.path.join(PLOTS_DIR, 'autoencoder_training_loss.png')
        plt.savefig(loss_plot, bbox_inches='tight', dpi=150)
        plt.close()

        # 7. MLflow logging
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("threshold", threshold)
        mlflow.log_metric("val_f1_best", val_f1)
        mlflow.log_metric("val_precision", threshold_metrics['precision'])
        mlflow.log_metric("val_recall", threshold_metrics['recall'])
        mlflow.log_artifact(eval_plot)
        mlflow.log_artifact(loss_plot)

        # 8. Save artifacts
        model_path = os.path.join(MODELS_DIR, 'autoencoder_model.keras')
        model.save(model_path)

        threshold_data = {
            "threshold": float(threshold),
            "validation_f1": float(val_f1),
            "validation_precision": float(threshold_metrics['precision']),
            "validation_recall": float(threshold_metrics['recall']),
            "method": "f1_optimization_on_validation"
        }
        with open(os.path.join(MODELS_DIR, 'autoencoder_threshold.json'), 'w') as f:
            json.dump(threshold_data, f, indent=2)

        meta = {
            **metrics,
            "model_name": "Autoencoder",
            "algorithm_type": "unsupervised",
            "architecture": "Dense(32)â†’Dense(16)â†’Dense(8)â†’Dense(16)â†’Dense(32)â†’Dense(30)",
            "training_time": training_time,
            "threshold": float(threshold),
            "validation_f1": float(val_f1),
            "validation_precision": float(threshold_metrics['precision']),
            "validation_recall": float(threshold_metrics['recall']),
            "error_metric": "combined(0.5*MSE+0.3*MAE+0.2*MaxErr)",
            "data_split": "train_70_val_15_test_15",
            "no_data_leakage": True,
            "mlflow_run_id": mlflow.active_run().info.run_id
        }
        
        with open(os.path.join(MODELS_DIR, 'autoencoder_metrics.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\n  âœ… Model saved: {model_path}")
        print(f"  âœ… Threshold saved: autoencoder_threshold.json")
        print(f"  âœ… Metrics saved: autoencoder_metrics.json")
        print(f"  âœ… MLflow Run ID: {meta['mlflow_run_id']}")
        print("\n" + "ðŸŸ£ " * 25)

        return model, meta, threshold


if __name__ == "__main__":
    train_autoencoder()
