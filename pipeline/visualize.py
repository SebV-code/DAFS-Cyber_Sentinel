import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, precision_recall_curve
)

def _ensure(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def make_plots(model, X_test, y_test, df_full, outdir: str):
    """
    Save PNGs: confusion matrix, ROC, PR curve, top feature importances (if available).
    TODO (Student E): add per-flag/per-service breakdowns, error analysis, and threshold-sweep plots.
    """
    _ensure(outdir)

    # Predictions / scores
    y_pred = model.predict(X_test)
    if hasattr(model.named_steps["model"], "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["normal (0)", "anomaly (1)"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "roc_curve.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    # Precision–Recall curve
    try:
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = auc(rec, prec)  # rough area; AP is in metrics.json => correction
        plt.figure()
        plt.plot(rec, prec, label=f"Area ≈ {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "pr_curve.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    # Feature importances (best-effort): try XGB feature names after preprocessing
    try:
        import numpy as np
        model_step = model.named_steps["model"]
        booster = model_step.get_booster()
        scores = booster.get_fscore()  # dict of {'f0': count, 'f1': count, ...}

        # Map fN to transformed feature indices
        # NOTE: ColumnTransformer + OHE expands columns; getting exact names is advanced.
        # As a simple proxy, plot top indices by importance:
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
        if items:
            names = [k for k,_ in items]
            vals  = [v for _,v in items]
            plt.figure(figsize=(6, max(3, len(items)*0.25)))
            sns.barplot(x=vals, y=names, orient="h")
            plt.title("Top XGBoost Split Importances (by feature index)")
            plt.xlabel("Split Count")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "xgb_feature_importance.png"), dpi=150)
            plt.close()
    except Exception:
        pass

     # add per-flag/per-service breakdowns
    for col in ["flag", "service"]: # on crée 2 images, une sur flag et l'autre sur service
        try:
            df_err = pd.DataFrame({ # création d'un df 3 col : service, vrai réponse, réponse du modèle
                col: X_test[col].values,
                "y_true": y_test.values,
                "y_pred": y_pred,
            })
            df_err["erreur"] = (df_err["y_true"] != df_err["y_pred"]).astype(int) # 4ème col : 1 si le modèle se trompe, 0 sinon

            stats = (df_err.groupby(col)["erreur"] # rangement des lignes par service
                           .agg(["mean", "count"]) # pourcentage d'erreur et nb de connexion pour chaque service
                           .sort_values("count", ascending=False)
                           .head(15)               # garde les 15 services les plus fréquents
                           .reset_index())         # remet le nom du service dans une vrai colonne

            plt.figure(figsize=(7, max(3, len(stats) * 0.35))) # préparatio de l'image vide et haute s'il y a beaucoup de barres
            sns.barplot(data=stats, x="mean", y=col, orient="h") # barre horizontale par service, longueur = % d'erreur
            plt.title(f"Taux d'erreur par {col} (top {len(stats)} par volume)")
            plt.xlabel("Taux d'erreur")
            plt.tight_layout()          # évite que le texte soit coupé
            plt.savefig(os.path.join(outdir, f"errors_by_{col}.png"), dpi=150) # sauvegarde de l'image dans outputs
            plt.close()
        except Exception as e:
            print(f"Erreur graphique {col} :", e)

    # Error analysis
    try:
        df_res = X_test.copy()              # copie pour pas abier l'original
        df_res["y_true"] = y_test.values    # ajout d'une colonne avec la vrai réponse (0=normal, 1=attaque)
        df_res["y_pred"] = y_pred           # ajout d'une colonne avec la réponse du modèle
        df_res["proba_anomalie"] = y_proba  # ajout d'une colonne avec la certitude du modèle

        df_res["resultat"] = "correct"      # création d'une colonne résultat avec la valeur correct
        # écrire fausse alerte ou attaque ratee uniquement sur les lignes respectant la condition
        df_res.loc[(df_res["y_true"] == 0) & (df_res["y_pred"] == 1), "resultat"] = "fausse alerte (FP)"
        df_res.loc[(df_res["y_true"] == 1) & (df_res["y_pred"] == 0), "resultat"] = "attaque ratee (FN)"

        erreurs = df_res[df_res["resultat"] != "correct"] # garder uniquement les lignes pas correctes
        erreurs.to_csv(os.path.join(outdir, "erreurs.csv"), index=False) # enregistre ces lignes dans le csv

        comptes = erreurs["resultat"].value_counts() # nb de FP et FN
        plt.figure(figsize=(6, 4)) # préparation image vide 6 x 4
        sns.barplot(x=comptes.index, y=comptes.values) # bar par type d'erreur
        plt.title(f"Types d'erreurs ({len(erreurs)} au total)")
        plt.ylabel("Nombre")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "types_erreurs.png"), dpi=150)
        plt.close()
    except Exception as e:
        print("Erreur analyse des erreurs :", e)

    # threshold-sweep plots
    # résultat du visu : 0.5 de seuil semble être le plus adapté car il vaut mieux rater peu de vrai attaques avec plus de fausses alertes
    try:
        seuils = np.linspace(0.05, 0.95, 19) # liste des seuils à tester
        nb_fp = []
        nb_fn = []
        for s in seuils:
            pred_s = (y_proba >= s).astype(int) # refais les prédictions avec le seuil
            nb_fp.append(int(((y_test.values == 0) & (pred_s == 1)).sum())) # compte les fausses alertes FP pour le seuil
            nb_fn.append(int(((y_test.values == 1) & (pred_s == 0)).sum())) # compte les alertes ratés FN pour le seuil

        plt.figure(figsize=(7, 4))
        plt.plot(seuils, nb_fp, marker="o", label="Fausses alertes (FP)") # courbe des FP
        plt.plot(seuils, nb_fn, marker="o", label="Attaques ratees (FN)") # courbe des FN
        plt.axvline(0.5, linestyle="--", color="grey", label="Seuil actuel (0.5)")
        plt.xlabel("Seuil de decision")
        plt.ylabel("Nombre d'erreurs")
        plt.title("Erreurs selon le seuil de decision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "threshold_sweep.png"), dpi=150)
        plt.close()
    except Exception as e:
        print("Erreur threshold sweep :", e)