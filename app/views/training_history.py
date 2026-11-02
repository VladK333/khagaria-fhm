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
