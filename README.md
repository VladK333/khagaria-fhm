# Khagaria flood susceptibility and hazard mapping

Code, data and results for **"Flood Susceptibility and Hazard Zonation of Khagaria District, North Bihar, India: A Machine Learning Approach Using Multi-Mission SAR Data"**, submitted to the *Journal of Earth System Science* (JESS-D-26-00905).

> **Use the v2 files** (`data/predictors_30m_v2/`, `models_v2/`, `results_v2/`, `manuscript/manuscript_v2.tex`). v2 is the re-analysis of October 2026. The files of the submitted version (19 Aug 2026) are kept for reference. What changed and why is listed below.

## What the study does
A SAR-derived flood inventory (union of the 2008, 2020 and 2021 events; ENVISAT ASAR and Sentinel-1) labels every 30 m pixel of the district as flood or non-flood. Five models (RandomForest, XGBoost, CatBoost, ANN, DNN) are trained on **14 conditioning factors** to give flood-susceptibility probability maps. Each map is turned into a binary hazard map with the threshold that maximises the Critical Success Index (**Max-CSI**), which is compared with the Youden-J threshold. SHAP explains the models.

## Why v2 (audit of October 2026)
| Problem in the submitted version | v2 fix |
|---|---|
| `Slope_30m.tif` (and `predictors_original/Slope.tif`) is an exact copy of `TWI_30m.tif`. The paper's "infinite VIF between Slope and TWI" was this duplicate. | Slope recomputed from `Elevation_30m` (Horn, degrees). Maximum VIF is now 1.82, so all 14 factors go to every model. |
| The label raster had no NoData, so 1.78 M pixels **outside** the district (fill values, e.g. Elevation −128) were training data labelled non-flood. AUC was inflated (ANN 0.951 → 0.835 inside the district). | Everything outside the district is masked; 1,632,234 district pixels are used. |
| XGBoost early stopping used the test set, and thresholds were picked on the full raster (training pixels included). | Split 70 / 10 / 20 (train / validation / test, seed 42). Early stopping and thresholds use the validation split only; every reported metric is on the test split. |
| The class-area table and "22.9 % Very High" were percentages of the whole rectangle (3,074 km²), not the district. | Areas are computed over the district (1,469 km²). |
| The Fig. 9 caption swapped the CatBoost/RF and ANN/DNN flood areas. The SHAP CSV covered only 2 models and did not match the paper's tables. | Regenerated from the v2 outputs. |
| Soil had 17,803 district pixels with fill code 65535 (248 of them in the 20k training table). | Filled with the nearest valid soil class. |

## Headline v2 results (independent test set, Max-CSI threshold from the validation set): `results_v2/metrics/test_metrics_maxcsi.csv`
| Model | τ* | AUC | CSI | Sensitivity | Specificity | Hazard area (km²) |
|---|---|---|---|---|---|---|
| XGBoost | 0.493 | 0.992 | 0.923 | 0.959 | 0.950 | 822.0 |
| CatBoost | 0.494 | 0.979 | 0.873 | 0.932 | 0.914 | 824.8 |
| RandomForest | 0.487 | 0.971 | 0.848 | 0.921 | 0.891 | 827.6 |
| DNN | 0.446 | 0.951 | 0.804 | 0.909 | 0.834 | 856.4 |
| ANN | 0.429 | 0.945 | 0.795 | 0.913 | 0.810 | 875.6 |

- **Inventory:** 824.4 km² flooded = 56.1 % of the 1,469 km² modelled district.
- **Max-CSI vs Youden:** Max-CSI wins for all five models, by +0.002 to +0.008 CSI. Max-CSI thresholds are lower than Youden's.
- **Calibration:** the neural networks are the best calibrated (ECE 0.013–0.014, against 0.023–0.054 for the trees). The submitted text called them "miscalibrated", which no longer holds.
- **SHAP:** Rainfall, Geomorphology, Soil and Elevation are ranks 1–4 in all five models; TWI, Slope, Aspect and Curvature are ranks 11–14.
- **Model agreement:** all five models give the same class on 85.4 % of the district.
- **Soil artefact:** the straight east–west line visible in the maps is a unit boundary in the ~1 km soil layer.

## Scripts in this repo
Only the plotting scripts are kept here; everything they read is already in `results_v2/`, `data/` and `models_v2/`.
```bash
conda env create -f environment_v2.yml && conda activate khagaria-fhm-v2
python scripts/v2/03_figures.py      # manuscript figures → results_v2/figures/
python scripts/v2/04_schematics.py   # flowchart + graphical abstract
```
`scripts/v2/nets.py` defines the ANN/DNN architectures, needed to load `models_v2/model_ann.pth` and `model_dnn.pth`. `scripts/visualization/` holds the plotting scripts of the submitted version (absolute paths from the author's machine).

The data-preparation, training and SHAP scripts were removed from this repository for now; they are available from the repository owner on request.

## Interactive app
A Streamlit app for exploring the v2 results in the browser. It only reads the CSV/JSON/NPY files in `results_v2/` and the pre-rendered map layers in `app/assets/maps/`; nothing is retrained. It needs four light packages, not the full analysis environment.
```bash
pip install -r app/requirements.txt
streamlit run app/main.py            # opens http://localhost:8501
```
| Page | What it shows |
|---|---|
| Model comparison | test metrics at the Max-CSI, Youden and 0.5 thresholds; AUC on train/val/test |
| Threshold explorer | CSI, sensitivity, specificity and confusion counts as the threshold moves, with Max-CSI and Youden marked |
| ROC and calibration | ROC curves per split and per model; reliability diagrams with ECE |
| Training history | XGBoost/CatBoost AUC per round, ANN/DNN loss and AUC per epoch |
| SHAP | global importance, normalised comparison and rank heatmap across models, beeswarm, dependence plots |
| Input data | correlation matrix and VIF of the 14 factors |
| Area statistics | area per susceptibility class and model agreement |
| Maps | probability, risk classes, binary hazard, errors vs the flood inventory (TP/FP/FN/TN) and model agreement on an OpenStreetMap basemap; two models side by side |
| Input factors | the 14 conditioning factors on the same basemap, with their SHAP rank and VIF |

The map pages show PNG layers in `app/assets/maps/`, made from the GeoTIFFs by `scripts/viz/prepare_map_layers.py` (reprojected to WGS84, coloured, downsampled to ~60 m). They are committed, so the app does not need rasterio. Re-run the script only if the rasters change:
```bash
pip install rasterio matplotlib pillow pyshp
python scripts/viz/prepare_map_layers.py
```
The error map covers every district pixel, training pixels included, so it flatters models that overfit; the test-split metrics on the other pages are the honest scores.

## Layout (v2)
| Path | What |
|---|---|
| `data/predictors_30m_v2/` | 14 aligned factor rasters, EPSG:32645, 30 m, 1769 × 1935, NoData −9999 outside the district |
| `data/inventory/label_30m_v2.tif` | label on the same grid: 1 flood, 0 non-flood, 255 outside the district |
| `data/district_mask_30m.tif` | 1 = modelled district pixel |
| `models_v2/` | `model_xgb.json`, `model_cat.cbm`, `model_rf.joblib` (scikit-learn), `model_ann.pth` / `model_dnn.pth` (architectures in `scripts/v2/nets.py`), `scaler.pkl`. Feature order: `results_v2/run_config.json` → `features` |
| `results_v2/metrics/` | test metrics at Max-CSI / Youden / 0.5, AUC train/val/test, VIF, correlation matrix, area statistics, natural breaks, model agreement |
| `results_v2/curves/` | ROC (train/val/test), CSI/sensitivity/specificity vs threshold (test), calibration tables: per model, as CSV |
| `results_v2/shap/` | mean \|SHAP\| and ranks (`mean_abs_shap.csv`), per-pixel SHAP values (`shap_values_<model>.npy`, 5000 × 14) and their feature values (`shap_sample_features.csv`) |
| `results_v2/training/` | training histories (XGBoost/CatBoost AUC per round, ANN/DNN loss per epoch), `sample_20k_pixels.csv` (balanced pixels with features, label, split, coordinates and all five predictions), `split_indices.npz` |
| `results_v2/maps/` | `probability/Map_*.tif`, `binary/Binary_*_MaxCSI.tif`, `binary/Agreement_count_MaxCSI.tif` (0–5 models), `classified/Classified_*_NaturalBreaks.tif` |
| `results_v2/figures/` | all manuscript figures |
| `app/` | interactive Streamlit app: `main.py` (entry point), `data.py` (cached loaders), `mapping.py` (shared map drawing), `views/` (one file per page), `assets/maps/` (pre-rendered map layers) |
| `scripts/viz/` | `prepare_map_layers.py`: builds `app/assets/maps/` from the result and factor GeoTIFFs |
| `manuscript/manuscript_v2.tex` / `.pdf` | revised manuscript (built by `make_v2.py` from the submitted `manuscript.tex`) |

### Submitted version (kept for reference)
`data/predictors_30m/`, `data/training_tables/` (20k points), `models/`, `results/` and `manuscript/manuscript.tex` are the 19 Aug 2026 version.

See `docs/VISUALIZATION_GUIDE.md` for the interactive-visualisation plan.
