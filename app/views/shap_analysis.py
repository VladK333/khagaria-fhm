import plotly.express as px
import streamlit as st

import data

UNITS = {"RandomForest": "probability"}

st.title("SHAP")
st.caption(
    "SHAP values show how much each conditioning factor pushes a pixel's prediction towards flood "
    "(positive) or non-flood (negative). Computed on a sample of 5,000 pixels."
)

model = st.selectbox("Model", data.MODELS)

st.subheader(f"Global importance: {model}")
categorical = set(data.run_config()["categorical_coded"])
imp = data.mean_abs_shap()
imp = imp[imp["Model"] == model].sort_values("mean_abs_SHAP")
imp["Type"] = ["Categorical" if f in categorical else "Continuous" for f in imp["Feature"]]
fig = px.bar(imp, x="mean_abs_SHAP", y="Feature", color="Type", orientation="h",
             color_discrete_map={"Continuous": "#534AB7", "Categorical": "#1D9E75"},
             hover_data={"rank": True, "Type": False},
             category_orders={"Feature": imp["Feature"].tolist()[::-1]})
fig.update_layout(xaxis_title=f"Mean |SHAP| ({UNITS.get(model, 'log-odds')})", yaxis_title="",
                  height=480, margin=dict(t=20), legend_title_text="")
st.plotly_chart(fig, width="stretch")
st.caption(
    "Mean absolute SHAP value over the 5,000 pixels: the average size of each factor's effect, "
    "regardless of direction. Units differ between models, so compare ranks rather than raw values across models."
)

st.subheader("Global importance: all models")
share = data.mean_abs_shap().copy()
share["Share (%)"] = 100 * share["mean_abs_SHAP"] / share.groupby("Model")["mean_abs_SHAP"].transform("sum")
# Ascending, because the y axis below is reversed so that XGBoost sits on top within each group
order = share.groupby("Feature")["Share (%)"].mean().sort_values().index.tolist()
fig = px.bar(share, x="Share (%)", y="Feature", color="Model", orientation="h", barmode="group",
             category_orders={"Feature": order, "Model": data.MODELS}, hover_data={"rank": True})
fig.update_yaxes(autorange="reversed")
fig.update_layout(xaxis_title="Share of the model's total mean |SHAP| (%)", yaxis_title="",
                  height=700, margin=dict(t=20), legend_title_text="")
st.plotly_chart(fig, width="stretch")
st.caption(
    "Each model's importances are scaled to sum to 100 %, which makes the tree models (log-odds and "
    "probability units) and the neural networks comparable. Factors are sorted by their average share."
)

st.subheader("Importance rank per model")
ranks = data.mean_abs_shap().pivot(index="Feature", columns="Model", values="rank")[data.MODELS]
ranks = ranks.loc[ranks.mean(axis=1).sort_values().index]
fig = px.imshow(ranks, text_auto=True, color_continuous_scale="Purples_r", aspect="auto",
                labels=dict(color="Rank"))
fig.update_layout(xaxis_title="", yaxis_title="", height=520, margin=dict(t=20))
fig.update_xaxes(side="top")
st.plotly_chart(fig, width="stretch")
st.caption(
    "1 = most important. Rainfall, Geomorphology, Soil and Elevation are in the top 4 for all five models; "
    "TWI, Slope, Aspect and Curvature are in the bottom 4."
)
