import plotly.graph_objects as go
import streamlit as st

import data

st.title("Training history")
st.caption(
    "How the validation score evolved during training. Early stopping and threshold selection "
    "used the validation split only; the test split was never seen during training."
)
st.info("RandomForest has no training history: its trees are grown independently, not iteratively.")

st.subheader("Gradient boosting: AUC per round")
xgb = data.training_history("XGBoost")
cat = data.training_history("CatBoost")
fig = go.Figure()
fig.add_trace(go.Scatter(x=xgb["round"], y=xgb["train_auc"], name="XGBoost train",
                         line=dict(color="#AFA9EC", width=1.5)))
fig.add_trace(go.Scatter(x=xgb["round"], y=xgb["val_auc"], name="XGBoost validation",
                         line=dict(color="#534AB7", width=2)))
fig.add_trace(go.Scatter(x=cat["iteration"], y=cat["val_auc"], name="CatBoost validation",
                         line=dict(color="#1D9E75", width=2)))
fig.update_layout(xaxis_title="Boosting round", yaxis_title="AUC", hovermode="x unified",
                  margin=dict(t=20), legend=dict(x=0.6, y=0.1))
st.plotly_chart(fig, width="stretch")
st.caption(
    "XGBoost ran the full 1,000 rounds: validation AUC was still rising, so early stopping (20 rounds "
    "without improvement) never triggered. CatBoost was trained for a fixed 800 iterations and logged "
    "validation AUC only."
)

st.subheader("Neural networks: loss and AUC per epoch")
NETS = {"ANN": "#D85A30", "DNN": "#1D9E75"}
left, right = st.columns(2)

fig = go.Figure()
for name, color in NETS.items():
    h = data.training_history(name)
    fig.add_trace(go.Scatter(x=h["epoch"], y=h["train_loss"], name=f"{name} train",
                             line=dict(color=color, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_loss"], name=f"{name} validation",
                             line=dict(color=color, width=2)))
fig.update_layout(title="Loss (binary cross-entropy)", xaxis_title="Epoch", yaxis_title="Loss",
                  hovermode="x unified", margin=dict(t=40), legend=dict(orientation="h", y=-0.25))
left.plotly_chart(fig, width="stretch")

fig = go.Figure()
for name, color in NETS.items():
    h = data.training_history(name)
    fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_auc"], name=f"{name} validation",
                             line=dict(color=color, width=2)))
fig.update_layout(title="Validation AUC", xaxis_title="Epoch", yaxis_title="AUC",
                  hovermode="x unified", margin=dict(t=40), legend=dict(orientation="h", y=-0.25))
right.plotly_chart(fig, width="stretch")
st.caption(
    "Both networks were trained for 15 epochs (Adam, learning rate 0.001, batch size 4,096). "
    "Validation loss below training loss is expected here: dropout is active during training only."
)
