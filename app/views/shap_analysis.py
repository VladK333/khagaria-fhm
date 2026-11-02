import streamlit as st

import data

st.title("SHAP")
st.caption(
    "SHAP values show how much each conditioning factor pushes a pixel's prediction towards flood "
    "(positive) or non-flood (negative). Computed on a sample of 5,000 pixels."
)

model = st.selectbox("Model", data.MODELS)
