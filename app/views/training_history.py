import streamlit as st

import data

st.title("Training history")
st.caption(
    "How the validation score evolved during training. Early stopping and threshold selection "
    "used the validation split only; the test split was never seen during training."
)
st.info("RandomForest has no training history: its trees are grown independently, not iteratively.")
