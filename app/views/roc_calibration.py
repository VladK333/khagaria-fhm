import plotly.graph_objects as go
import streamlit as st

import data
from widgets import pick

SPLITS = {"train": ("Train", "#AFA9EC"), "val": ("Validation", "#534AB7"), "test": ("Test", "#D85A30")}

st.title("ROC and calibration")
st.caption(
    "ROC curves show how well each model separates flood from non-flood pixels at every threshold. "
    "Calibration shows whether a predicted probability of, say, 0.7 really means a 70 % flood frequency."
)

model = pick("Model", data.MODELS, "XGBoost", key="roc_model")

st.subheader(f"ROC curves: {model}")
auc = data.auc_splits().set_index("Model").loc[model]
fig = go.Figure()
for split, (label, color) in SPLITS.items():
    roc = data.roc_curve(model, split)
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], name=f"{label} (AUC {auc[f'{split}_AUC']:.3f})",
                             line=dict(color=color, width=2)))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="gray", dash="dot")))
fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                  xaxis_range=[0, 1], yaxis_range=[0, 1.02], margin=dict(t=20),
                  legend=dict(x=0.55, y=0.08))
st.plotly_chart(fig, width="stretch")
st.caption("Train, validation and test curves lying on top of each other mean the model generalises well.")

st.subheader("ROC curves: all models (test)")
test_auc = data.auc_splits().set_index("Model")["test_AUC"]
fig = go.Figure()
for name in data.MODELS:
    roc = data.roc_curve(name, "test")
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], name=f"{name} (AUC {test_auc[name]:.3f})",
                             line=dict(width=3 if name == model else 1.5)))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="gray", dash="dot")))
fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                  xaxis_range=[0, 1], yaxis_range=[0, 1.02], margin=dict(t=20),
                  legend=dict(x=0.55, y=0.08))
st.plotly_chart(fig, width="stretch")
st.caption("The selected model is drawn thicker. Zoom into the top-left corner to see the differences.")

st.subheader(f"Calibration: {model}")
cal = data.calibration(model)
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", line=dict(color="gray", dash="dot")))
fig.add_trace(go.Scatter(x=cal["mean_pred"], y=cal["obs_freq"], name=model, mode="lines+markers",
                         line=dict(color="#534AB7", width=2), customdata=cal["n"],
                         hovertemplate="Predicted %{x:.3f}<br>Observed %{y:.3f}<br>%{customdata:,} pixels"))
fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed flood frequency",
                  xaxis_range=[0, 1], yaxis_range=[0, 1.02], margin=dict(t=20),
                  legend=dict(x=0.02, y=0.98))
st.plotly_chart(fig, width="stretch")
st.caption(
    "Each point is one of 10 probability bins on the test set. Points on the diagonal mean the predicted "
    "probability matches how often those pixels were actually flooded. Below the diagonal the model "
    "over-predicts flooding, above it the model under-predicts."
)

st.subheader("Calibration: all models (test)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", line=dict(color="gray", dash="dot")))
for name in data.MODELS:
    cal = data.calibration(name)
    ece = ((cal["mean_pred"] - cal["obs_freq"]).abs() * cal["n"]).sum() / cal["n"].sum()
    fig.add_trace(go.Scatter(x=cal["mean_pred"], y=cal["obs_freq"], name=f"{name} (ECE {ece:.3f})",
                             mode="lines+markers", line=dict(width=3 if name == model else 1.5)))
fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed flood frequency",
                  xaxis_range=[0, 1], yaxis_range=[0, 1.02], margin=dict(t=20),
                  legend=dict(x=0.02, y=0.98))
st.plotly_chart(fig, width="stretch")
st.caption(
    "ECE (expected calibration error) is the pixel-weighted average distance from the diagonal; lower is better."
)
