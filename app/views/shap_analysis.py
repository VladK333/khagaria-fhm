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
