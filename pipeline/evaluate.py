import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, classification_report, brier_score_loss
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

    # Per-class metrics
    # but : dans metrics.json on a les metrics pour les attaques et on souhaite calculer pour les connexions normales
    report = classification_report(         # demande de bulletin de notes
        y_test, y_pred,                     # comparaison des vrais réponses aux réponses du modèle
        target_names=["normal", "anomaly"], # appelle la classe 0 normal et la classe 1 anomaly », au lieu de « 0 » et « 1 ».
        output_dict=True, zero_division=0,  # donne-moi le bulletin sous forme de dictionnaire, si un calcul divise par zéro, mets 0 au lieu de planter
    )
    for cls in ["normal", "anomaly"]:       # applique 2 fois : pour normal et anomaly
        # va chercher la précision de cette classe dans le bulletin et range la dans metrics
        metrics[f"{cls}_precision"] = float(report[cls]["precision"]) 
        metrics[f"{cls}_recall"] = float(report[cls]["recall"])
        metrics[f"{cls}_f1"] = float(report[cls]["f1-score"])
        metrics[f"{cls}_support"] = int(report[cls]["support"])

    # Threshold search (best F1 + recall target)
    target_recall = 0.99                    # objectif : detecter au moins 99% des attaques
    best_f1, best_f1_threshold = 0.0, 0.5   
    recall_threshold = None
    for t in np.linspace(0.01, 0.99, 99):   # teste les seuils un par un 0.01, 0.02, 0.03...0.99
        pred_t = (y_proba >= t).astype(int) # refais les prédictions avec ce seuil : 1 (probabilité au dessus) 0 sinon
        f1_t = f1_score(y_test, pred_t, zero_division=0) # calcul de f1 avec ce seuil
        if f1_t > best_f1:                               # si la note bat le record, retiens la avec ce seuil
            best_f1, best_f1_threshold = f1_t, t
        if recall_score(y_test, pred_t, zero_division=0) >= target_recall: # Si le seuil trouve au moins 99% des attaques, retiens-le
            recall_threshold = t
    metrics["best_f1"] = float(best_f1)  # range le meilleur f1
    metrics["best_f1_threshold"] = round(float(best_f1_threshold), 2)  # range le seuil arrondi 
    metrics["target_recall"] = target_recall  # note l'objectif fixé au départ
    metrics["threshold_for_target_recall"] = round(float(recall_threshold), 2) if recall_threshold is not None else None

    return metrics
