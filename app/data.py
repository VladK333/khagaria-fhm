"""Loaders for the v2 outputs. Every function only reads files under results_v2/."""
import json
from pathlib import Path

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
