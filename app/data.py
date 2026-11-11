"""Loaders for the v2 outputs. Every function only reads files under results_v2/."""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_v2"
MAP_LAYERS = Path(__file__).resolve().parent / "assets" / "maps"  # built by scripts/viz/prepare_map_layers.py

MODELS = ["XGBoost", "CatBoost", "RandomForest", "DNN", "ANN"]


class MissingDataError(FileNotFoundError):
    """A file the app needs is not there; carries a hint on how to get it back (shown by main.py)."""

    def __init__(self, path: Path):
        self.path = path
        if MAP_LAYERS in path.parents:
            self.hint = "Map layers are built by `python scripts/viz/prepare_map_layers.py` (see the README)."
        else:
            self.hint = ("This file is part of the project results, which are committed to the repository. "
                         "Check that the repository was cloned completely (a ZIP download may miss files).")
        super().__init__(f"Missing file: {path.relative_to(ROOT).as_posix()}")


def _file(path: Path) -> Path:
    """The path itself if the file exists, otherwise MissingDataError."""
    if not path.is_file():
        raise MissingDataError(path)
    return path


@st.cache_data
def run_config() -> dict:
    with open(_file(RESULTS / "run_config.json")) as f:
        return json.load(f)


@st.cache_data
def test_metrics() -> pd.DataFrame:
    """Test-set metrics at the Max-CSI, Youden and 0.5 thresholds."""
    return pd.read_csv(_file(RESULTS / "metrics" / "test_metrics_all_thresholds.csv"))


@st.cache_data
def auc_splits() -> pd.DataFrame:
    """AUC on the train, validation and test splits."""
    return pd.read_csv(_file(RESULTS / "metrics" / "auc_train_val_test.csv"))


@st.cache_data
def area_statistics() -> pd.DataFrame:
    """Area per susceptibility class (km2 and % of the district) and Max-CSI hazard area per model."""
    return pd.read_csv(_file(RESULTS / "metrics" / "area_statistics.csv"))


@st.cache_data
def agreement_areas() -> pd.DataFrame:
    """Area (km2) by the number of models (0-5) that flag a pixel as flood at their Max-CSI threshold."""
    return pd.read_csv(_file(RESULTS / "metrics" / "agreement_areas.csv"))


@st.cache_data
def threshold_curve(model: str) -> pd.DataFrame:
    """Test-set CSI, sensitivity, specificity and confusion counts per threshold (step 0.005)."""
    return pd.read_csv(_file(RESULTS / "curves" / f"threshold_curve_{model}_test.csv"))


@st.cache_data
def roc_curve(model: str, split: str) -> pd.DataFrame:
    """ROC curve (fpr, tpr, threshold) for one model on the train, val or test split."""
    return pd.read_csv(_file(RESULTS / "curves" / f"roc_{model}_{split}.csv"))


@st.cache_data
def calibration(model: str) -> pd.DataFrame:
    """Reliability table on the test split: 10 bins of mean predicted vs observed flood frequency."""
    return pd.read_csv(_file(RESULTS / "curves" / f"calibration_{model}_test.csv"))


@st.cache_data
def training_history(model: str) -> pd.DataFrame:
    """Per-round (XGBoost, CatBoost) or per-epoch (ANN, DNN) training history."""
    return pd.read_csv(_file(RESULTS / "training" / f"history_{model}.csv"))


@st.cache_data
def mean_abs_shap() -> pd.DataFrame:
    """Global importance: mean |SHAP| and rank per model and feature.
    RandomForest is in probability units, the other models in log-odds."""
    return pd.read_csv(_file(RESULTS / "shap" / "mean_abs_shap.csv"))


@st.cache_data
def shap_values(model: str) -> pd.DataFrame:
    """Per-pixel SHAP values (5,000 pixels x 14 features), rows aligned with shap_features()."""
    values = np.load(_file(RESULTS / "shap" / f"shap_values_{model}.npy"))
    return pd.DataFrame(values, columns=run_config()["features"])


@st.cache_data
def correlation_matrix() -> pd.DataFrame:
    """Pairwise correlation between the 14 conditioning factors."""
    return pd.read_csv(_file(RESULTS / "metrics" / "correlation_matrix.csv"), index_col=0)


@st.cache_data
def vif() -> pd.DataFrame:
    """Variance inflation factor per conditioning factor."""
    return pd.read_csv(_file(RESULTS / "metrics" / "VIF.csv"))


@st.cache_data
def map_layer(name: str) -> tuple[str, list[list[float]]]:
    """A pre-rendered map layer as a PNG data URI plus its corner coordinates (lon, lat)."""
    png = base64.b64encode(_file(MAP_LAYERS / f"{name}.png").read_bytes()).decode()
    with open(_file(MAP_LAYERS / f"{name}.json")) as f:
        corners = json.load(f)["coordinates"]
    return f"data:image/png;base64,{png}", corners


@st.cache_data
def map_legend(name: str) -> dict:
    """Legend stored next to a factor layer: vmin/vmax/colorscale, or class code -> colour."""
    with open(_file(MAP_LAYERS / f"{name}.json")) as f:
        return json.load(f)["legend"]


@st.cache_data
def error_counts_district() -> pd.DataFrame:
    """TP/FP/FN/TN pixels of each binary map over the whole district (training pixels included)."""
    return pd.read_csv(_file(MAP_LAYERS / "error_counts_district.csv"), index_col="Model")


@st.cache_data
def district_boundary() -> dict:
    """District boundary as WGS84 GeoJSON."""
    with open(_file(MAP_LAYERS / "district_boundary.geojson")) as f:
        return json.load(f)


@st.cache_data
def shap_features() -> pd.DataFrame:
    """Feature values, label and pixel index of the 5,000 pixels used for SHAP."""
    return pd.read_csv(_file(RESULTS / "shap" / "shap_sample_features.csv"))
