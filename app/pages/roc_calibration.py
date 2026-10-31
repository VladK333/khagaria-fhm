import streamlit as st

import data

st.title("ROC and calibration")
st.caption(
    "ROC curves show how well each model separates flood from non-flood pixels at every threshold. "
    "Calibration shows whether a predicted probability of, say, 0.7 really means a 70 % flood frequency."
)

model = st.selectbox("Model", data.MODELS)
