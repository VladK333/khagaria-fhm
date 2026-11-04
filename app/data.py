"""Loaders for the v2 outputs. Every function only reads files under results_v2/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_v2"

MODELS = ["XGBoost", "CatBoost", "RandomForest", "DNN", "ANN"]


@st.cache_data
def run_config() -> dict:
    with open(RESULTS / "run_config.json") as f:
        return json.load(f)


@st.cache_data
def test_metrics() -> pd.DataFrame:
    """Test-set metrics at the Max-CSI, Youden and 0.5 thresholds."""
    return pd.read_csv(RESULTS / "metrics" / "test_metrics_all_thresholds.csv")


@st.cache_data
def auc_splits() -> pd.DataFrame:
    """AUC on the train, validation and test splits."""
    return pd.read_csv(RESULTS / "metrics" / "auc_train_val_test.csv")


@st.cache_data
def threshold_curve(model: str) -> pd.DataFrame:
    """Test-set CSI, sensitivity, specificity and confusion counts per threshold (step 0.005)."""
    return pd.read_csv(RESULTS / "curves" / f"threshold_curve_{model}_test.csv")


@st.cache_data
def roc_curve(model: str, split: str) -> pd.DataFrame:
    """ROC curve (fpr, tpr, threshold) for one model on the train, val or test split."""
    return pd.read_csv(RESULTS / "curves" / f"roc_{model}_{split}.csv")


@st.cache_data
def calibration(model: str) -> pd.DataFrame:
    """Reliability table on the test split: 10 bins of mean predicted vs observed flood frequency."""
    return pd.read_csv(RESULTS / "curves" / f"calibration_{model}_test.csv")


@st.cache_data
def training_history(model: str) -> pd.DataFrame:
    """Per-round (XGBoost, CatBoost) or per-epoch (ANN, DNN) training history."""
    return pd.read_csv(RESULTS / "training" / f"history_{model}.csv")


@st.cache_data
def mean_abs_shap() -> pd.DataFrame:
    """Global importance: mean |SHAP| and rank per model and feature.
    RandomForest is in probability units, the other models in log-odds."""
    return pd.read_csv(RESULTS / "shap" / "mean_abs_shap.csv")


@st.cache_data
def shap_values(model: str) -> pd.DataFrame:
    """Per-pixel SHAP values (5,000 pixels x 14 features), rows aligned with shap_features()."""
    values = np.load(RESULTS / "shap" / f"shap_values_{model}.npy")
    return pd.DataFrame(values, columns=run_config()["features"])


@st.cache_data
def correlation_matrix() -> pd.DataFrame:
    """Pairwise correlation between the 14 conditioning factors."""
    return pd.read_csv(RESULTS / "metrics" / "correlation_matrix.csv", index_col=0)


@st.cache_data
def vif() -> pd.DataFrame:
    """Variance inflation factor per conditioning factor."""
    return pd.read_csv(RESULTS / "metrics" / "VIF.csv")


@st.cache_data
def shap_features() -> pd.DataFrame:
    """Feature values, label and pixel index of the 5,000 pixels used for SHAP."""
    return pd.read_csv(RESULTS / "shap" / "shap_sample_features.csv")
