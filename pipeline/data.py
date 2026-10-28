import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET_COL = "class"   # values: "normal" / "anomaly"
EXPECTED_N_COL = 42

def load_data(path: str = "data/train_data.csv") -> pd.DataFrame:
    """
    Load the intrusion dataset and standardize target to {0,1} with column name 'class'.
    1 = anomaly (positive class), 0 = normal.
    TODO (Student A): add dtype hints, NA handling policy, schema validation, and logging.
    """
    df = pd.read_csv(path, dtype={c: str for c in EXPECTED_CATEGORICAL})
    logger.info(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # basic sanity checks
    missing = [c for c in EXPECTED_CATEGORICAL if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected categorical columns: {missing}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"CSV missing target column '{TARGET_COL}'")

    if df.shape[1] != EXPECTED_N_COL:
        raise ValueError(f"expected {EXPECTED_N_COL} columns, got {df.shape[1]}")

    num_cols = [c for c in df.columns if c not in EXPECTED_CATEGORICAL and c != TARGET_COL]
    not_numeric = [c for c in num_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if not_numeric:
        raise ValueError(f"Non-numeric values in numeric columns: {not_numeric}")

    n_dup= df.duplicated().sum()
    if n_dup > 0:
        logger.warning(f"{n_dup} lignes en double reperées")
    constant_cols= [c for c in df.columns if df[c].nunique() == 1]
    if constant_cols:
        logger.warning (f"Colonnes constantes (une seule valeur) : {constant_cols}")

    na_counts = df.isna().sum()           # nombre de NA par colonne
    na_cols = na_counts[na_counts > 0]    # seulement les colonnes qui en ont
    if not na_cols.empty:
        logger.warning(f"Valeurs manquantes : {na_cols.to_dict()}")

    n_before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"{n_dropped} lignes supprimées : cible '{TARGET_COL}' manquante")

    df[EXPECTED_CATEGORICAL] = df[EXPECTED_CATEGORICAL].fillna("unknown")

    
    # Normalize target → binary 0/1
    mapping = {"anomaly": 1, "normal": 0}
    if pd.api.types.is_string_dtype(df[TARGET_COL]):
        df[TARGET_COL] = df[TARGET_COL].map(mapping)
    # If already numeric, assume 0/1
    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("Target must be 'normal'/'anomaly' or 0/1 after mapping.")
    logger.info (f"Repartitions des valeurs cibles: {(df[TARGET_COL]==0).sum()} connexions normales, {(df[TARGET_COL]==1).sum()} anomalies")
    return df
