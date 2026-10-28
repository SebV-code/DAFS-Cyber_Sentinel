from typing import Optional
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

def _scale_pos_weight(y):
    # ratio of negatives to positives (for imbalance)
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    return neg / pos

def train_model(X_train, y_train, preproc, seed: int = 42):
    """
    Build a Pipeline(preproc → XGBClassifier) and fit.
    Uses scale_pos_weight to handle class imbalance (anomaly = positive).
    TODO (Student C): tune hyperparams, early_stopping_rounds with a valid set, CV grid, threshold tuning.
     
    """
    X_tr, X_valid, y_tr, y_valid = train_test_split(
    X_train, y_train,
    test_size=0.2,      # la proportion qui part dans le 2e morceau
    stratify=y_train,       # sur quelle variable garder les proportions de classes
    random_state=seed,   # pour la reproductibilité
    )
    spw = _scale_pos_weight(y_tr)
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=spw,  # key for imbalance
        verbosity=0,
        use_label_encoder=False
    )
    pipe = Pipeline([("preproc", preproc), ("model", clf)])
    pipe.fit(X_tr, y_tr)
    return pipe
