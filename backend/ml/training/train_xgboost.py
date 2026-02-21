"""
FraudX Analyst - XGBoost Training (FIXED - No Leakage)
========================================================
Proper workflow:
1. Train on TRAIN set
2. Tune hyperparameters using VALIDATION set (Optuna)
3. Evaluate ONCE on TEST set
4. Report full metrics: accuracy, precision, recall, F1, AUC, confusion matrix, PR-AUC
"""

import os, json, time, warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import optuna
import shap
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix, average_precision_score,
                              precision_recall_curve, roc_curve)

from preprocess import load_and_preprocess

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models_saved')
PLOTS_DIR  = os.path.join(BASE_DIR, 'plots')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


def compute_full_metrics(y_true, y_pred, y_prob):
    """Compute comprehensive metrics including PR-AUC and confusion matrix."""
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


def make_objective(X_train, y_train, X_val, y_val, scale_pos_weight):
    """
    Optuna objective - TUNES ON VALIDATION SET ONLY.
    Never touches test set during hyperparameter search.
    """
    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
            "max_depth"        : trial.suggest_int("max_depth", 3, 9),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
            "gamma"            : trial.suggest_float("gamma", 0, 5),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 0, 2),
            "scale_pos_weight" : scale_pos_weight,
            "eval_metric"      : "logloss",
            "random_state"     : 42,
            "n_jobs"           : -1,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate on VALIDATION set (not test!)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, zero_division=0)

    return objective


def train_xgboost():
    print("\n" + "🔷 " * 25)
    print("  XGBoost Training (No Leakage - Validation Tuning)")
    print("🔷 " * 25)

    # 1. Load data with proper splits
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_preprocess()

    # Calculate class weight
    n_normal = (y_train == 0).sum()
    n_fraud  = (y_train == 1).sum()
    scale_pos_weight = round(n_normal / n_fraud, 2)
    
    print(f"\n  scale_pos_weight = {scale_pos_weight}")
    print(f"  (Fraud cases weighted {scale_pos_weight}x higher)")

    # 2. Optuna tuning on VALIDATION set
    print(f"\n  🔍 Optuna Tuning (50 trials on VALIDATION set)")
    print("  " + "-" * 56)

    mlflow.set_experiment("FraudX-Models-Fixed")

    with mlflow.start_run(run_name="XGBoost_NoLeakage"):
        
        t0 = time.time()

        study = optuna.create_study(direction="maximize")
        study.optimize(
            make_objective(X_train, y_train, X_val, y_val, scale_pos_weight),
            n_trials=50,
            show_progress_bar=True
        )

        best_params = study.best_params
        best_params["scale_pos_weight"] = scale_pos_weight
        best_params["eval_metric"] = "logloss"
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1

        print(f"\n  ✅ Best F1 on validation: {study.best_value:.4f}")
        print(f"  Best params: {best_params}")

        # 3. Train final model on train set with best params
        print("\n  🎯 Training final model with best params...")
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        training_time = round(time.time() - t0, 2)

        # 4. Evaluate on TEST set (FIRST AND ONLY TIME)
        print("\n  📊 Final Evaluation on TEST Set")
        print("  " + "-" * 56)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
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

        # 5. Plot confusion matrix + curves
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        im = axes[0,0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0,0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=axes[0,0])
        classes = ['Normal', 'Fraud']
        tick_marks = np.arange(len(classes))
        axes[0,0].set_xticks(tick_marks)
        axes[0,0].set_xticklabels(classes)
        axes[0,0].set_yticks(tick_marks)
        axes[0,0].set_yticklabels(classes)
        axes[0,0].set_ylabel('True Label', fontsize=12)
        axes[0,0].set_xlabel('Predicted Label', fontsize=12)
        
        thresh = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            axes[0,0].text(j, i, format(cm[i, j], 'd'),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black",
                          fontsize=14, fontweight='bold')
        
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
        
        # Feature Importance
        importance = model.feature_importances_
        indices = np.argsort(importance)[-15:]  # Top 15
        axes[1,1].barh(range(len(indices)), importance[indices], color='steelblue')
        axes[1,1].set_yticks(range(len(indices)))
        axes[1,1].set_yticklabels([feature_names[i] for i in indices], fontsize=10)
        axes[1,1].set_xlabel('Importance', fontsize=12)
        axes[1,1].set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
        axes[1,1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        metrics_plot = os.path.join(PLOTS_DIR, 'xgboost_evaluation.png')
        plt.savefig(metrics_plot, bbox_inches='tight', dpi=150)
        plt.close()

        # 6. SHAP
        print("\n  🔍 Computing SHAP values (500 samples)...")
        idx = np.random.choice(len(X_test), size=min(500, len(X_test)), replace=False)
        X_sample = X_test[idx]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        shap_plot = os.path.join(PLOTS_DIR, 'xgboost_shap_summary.png')
        plt.savefig(shap_plot, bbox_inches='tight', dpi=150)
        plt.close()

        # 7. MLflow logging
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("val_f1_best", study.best_value)
        mlflow.log_artifact(metrics_plot)
        mlflow.log_artifact(shap_plot)
        mlflow.xgboost.log_model(model, "xgboost_model")

        # 8. Save artifacts
        joblib.dump(model, os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
        joblib.dump(explainer, os.path.join(MODELS_DIR, 'xgboost_explainer.pkl'))

        meta = {
            **metrics,
            "model_name": "XGBoost",
            "algorithm_type": "supervised",
            "training_time": training_time,
            "scale_pos_weight": scale_pos_weight,
            "optuna_trials": 50,
            "validation_f1_best": round(study.best_value, 4),
            "best_params": best_params,
            "data_split": "train_70_val_15_test_15",
            "no_data_leakage": True,
            "mlflow_run_id": mlflow.active_run().info.run_id
        }
        
        with open(os.path.join(MODELS_DIR, 'xgboost_metrics.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\n  ✅ Model saved: {os.path.join(MODELS_DIR, 'xgboost_model.pkl')}")
        print(f"  ✅ Metrics saved: {os.path.join(MODELS_DIR, 'xgboost_metrics.json')}")
        print(f"  ✅ Plots saved: {PLOTS_DIR}")
        print(f"  ✅ MLflow Run ID: {meta['mlflow_run_id']}")
        print("\n" + "🔷 " * 25)

        return model, meta


if __name__ == "__main__":
    train_xgboost()
