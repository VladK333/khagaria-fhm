import plotly.graph_objects as go
import streamlit as st

import data

SPLITS = {"train": ("Train", "#AFA9EC"), "val": ("Validation", "#534AB7"), "test": ("Test", "#D85A30")}

st.title("ROC and calibration")
st.caption(
    "ROC curves show how well each model separates flood from non-flood pixels at every threshold. "
    "Calibration shows whether a predicted probability of, say, 0.7 really means a 70 % flood frequency."
)

model = st.selectbox("Model", data.MODELS)

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
st.plotly_chart(fig, use_container_width=True)
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
st.plotly_chart(fig, use_container_width=True)
st.caption("The selected model is drawn thicker. Zoom into the top-left corner to see the differences.")
