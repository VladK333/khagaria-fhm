import numpy as np
import plotly.express as px
import streamlit as st

import data

st.title("Input data")
st.caption(
    "The 14 conditioning factors used by every model. Strongly correlated factors carry the same information "
    "and make SHAP importances harder to interpret, so they were checked before training."
)

st.subheader("Correlation matrix")
corr = data.correlation_matrix()
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                labels=dict(color="r"))
fig.update_layout(height=620, margin=dict(t=20))
fig.update_xaxes(side="top", tickangle=-45)
st.plotly_chart(fig, width="stretch")

pair = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack().idxmax()
st.caption(
    f"Strongest pair: {pair[0]} and {pair[1]} (r = {corr.loc[pair]:.2f}). "
    "In the submitted version Slope was a copy of TWI (r = 1); v2 recomputes Slope from Elevation."
)
