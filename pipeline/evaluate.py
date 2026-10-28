import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss
)

def evaluate(model, X_test, y_test) -> dict:
    """
    Compute core binary classification metrics.
    IMPORTANT: anomaly = positive class.
    TODO (Student D): add calibration metrics, per-class metrics, threshold search for F1/recall targets.
    """
    y_pred = model.predict(X_test)
    if hasattr(model.named_steps["model"], "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: some estimators only have decision_function
        if hasattr(model.named_steps["model"], "decision_function"):
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            y_proba = y_pred.astype(float)
    brier_score = brier_score_loss(y_test, y_proba)   # Le score de Brier mesure l’écart entre les probabilités prédites et la réalité 0 = parfait

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) == 2 else None,
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "positives_test": int(np.sum(y_test == 1)),
        "negatives_test": int(np.sum(y_test == 0)),
        "brier_score": float(brier_score),
    }
    return metrics
