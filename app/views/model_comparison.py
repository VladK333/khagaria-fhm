import plotly.express as px
import streamlit as st

import data

METHODS = {"Max-CSI": "MaxCSI", "Youden": "Youden", "Fixed 0.5": "Fixed0.5"}
METRICS = ["CSI", "Sensitivity", "Specificity", "Accuracy", "Precision"]

st.title("Model comparison")
st.caption(
    "Independent test set (326,447 pixels). Thresholds are selected on the validation split."
)

method = st.segmented_control("Threshold", list(METHODS), default="Max-CSI")
if method is None:
    method = "Max-CSI"

df = data.test_metrics()
df = df[df["Threshold_method"] == METHODS[method]].set_index("Model").loc[data.MODELS]

best = df["CSI"].idxmax()
cols = st.columns(3)
cols[0].metric("Best model (CSI)", best)
cols[1].metric("CSI", f"{df.loc[best, 'CSI']:.3f}")
cols[2].metric("AUC", f"{df.loc[best, 'AUC']:.3f}")

long = df[METRICS].reset_index().melt(id_vars="Model", var_name="Metric", value_name="Value")
fig = px.bar(long, x="Metric", y="Value", color="Model", barmode="group",
             category_orders={"Model": data.MODELS})
fig.update_yaxes(range=[0.7, 1])
fig.update_layout(legend_title_text="", margin=dict(t=20))
st.plotly_chart(fig, width="stretch")

st.subheader("Metrics table")
table = df[["Threshold", "AUC"] + METRICS + ["TP", "FP", "FN", "TN"]]
st.dataframe(
    table.style.highlight_max(subset=["AUC"] + METRICS, color="rgba(29, 158, 117, 0.25)"),
    column_config={
        **{c: st.column_config.NumberColumn(format="%.3f") for c in ["Threshold", "AUC"] + METRICS},
        **{c: st.column_config.NumberColumn(format="localized") for c in ["TP", "FP", "FN", "TN"]},
    },
    width="stretch",
)

st.subheader("AUC on train, validation and test")
st.caption("Small gaps between the splits mean the models are not overfitting.")
auc = data.auc_splits().set_index("Model").loc[data.MODELS].reset_index()
auc = auc.rename(columns={"train_AUC": "Train", "val_AUC": "Validation", "test_AUC": "Test"})
fig = px.bar(auc.melt(id_vars="Model", var_name="Split", value_name="AUC"),
             x="Model", y="AUC", color="Split", barmode="group")
fig.update_yaxes(range=[0.9, 1])
fig.update_layout(legend_title_text="", margin=dict(t=20))
st.plotly_chart(fig, width="stretch")
