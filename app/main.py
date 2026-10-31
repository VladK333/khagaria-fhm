"""Entry point: streamlit run app/main.py"""
import streamlit as st

st.set_page_config(page_title="Khagaria FHM", page_icon=":ocean:", layout="wide")

pages = [
    st.Page("pages/model_comparison.py", title="Model comparison", icon=":material/bar_chart:", default=True),
    st.Page("pages/threshold_explorer.py", title="Threshold explorer", icon=":material/tune:"),
    st.Page("pages/roc_calibration.py", title="ROC and calibration", icon=":material/show_chart:"),
]

st.navigation(pages).run()
