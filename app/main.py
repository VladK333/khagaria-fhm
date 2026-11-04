"""Entry point: streamlit run app/main.py"""
import streamlit as st

st.set_page_config(page_title="Khagaria FHM", page_icon=":ocean:", layout="wide")

pages = [
    st.Page("views/model_comparison.py", title="Model comparison", icon=":material/bar_chart:", default=True),
    st.Page("views/threshold_explorer.py", title="Threshold explorer", icon=":material/tune:"),
    st.Page("views/roc_calibration.py", title="ROC and calibration", icon=":material/show_chart:"),
    st.Page("views/training_history.py", title="Training history", icon=":material/history:"),
    st.Page("views/shap_analysis.py", title="SHAP", icon=":material/insights:"),
    st.Page("views/input_data.py", title="Input data", icon=":material/table_chart:"),
    st.Page("views/area_statistics.py", title="Area statistics", icon=":material/area_chart:"),
]

st.navigation(pages).run()
