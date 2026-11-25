import plotly.express as px
import streamlit as st

import data

CLASSES = ["Very Low", "Low", "Moderate", "High", "Very High"]
COLORS = ["#E1F5EE", "#9FE1CB", "#FAC775", "#F0997B", "#D85A30"]

st.title("Area statistics")
st.caption("Areas are computed over the modelled district (1,469 km²), not the whole raster rectangle.")

st.subheader("Area per susceptibility class")
unit = st.segmented_control("Unit", ["%", "km²"], default="%")
if unit is None:
    unit = "%"
column = {"%": "{} (%)", "km²": "{} (km2)"}[unit]

areas = data.area_statistics().set_index("Model").loc[data.MODELS]
long = (areas[[column.format(c) for c in CLASSES]]
        .set_axis(CLASSES, axis=1).reset_index()
        .melt(id_vars="Model", var_name="Class", value_name=unit))
fig = px.bar(long, x=unit, y="Model", color="Class", orientation="h", text_auto=".0f",
             color_discrete_sequence=COLORS, category_orders={"Class": CLASSES, "Model": data.MODELS})
fig.update_layout(yaxis_title="", legend_title_text="", height=380, margin=dict(t=20),
                  legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, width="stretch")

breaks = data.run_config()["natural_breaks"]
st.caption(
    "Classes use the same natural breaks for every model, computed on the XGBoost probability map: "
    + ", ".join(f"{b:.3f}" for b in breaks)
    + ". XGBoost is the most decisive model (most pixels in Very Low or Very High); the neural networks "
    "spread more area over the middle classes."
)

st.subheader("Model agreement")
agree = data.agreement_areas()
district = agree["km2"].sum()
unanimous = agree.loc[agree["models_flagging"].isin([0, 5]), "km2"].sum()

cols = st.columns(3)
cols[0].metric("All models agree", f"{100 * unanimous / district:.1f} %")
cols[1].metric("Flood in all 5 models", f"{agree.set_index('models_flagging').loc[5, 'km2']:,.0f} km²")
cols[2].metric("Models disagree", f"{district - unanimous:,.0f} km²")

agree["Models flagging flood"] = agree["models_flagging"].astype(str)
fig = px.bar(agree, x="Models flagging flood", y="km2", text_auto=",.0f",
             color="Models flagging flood",
             color_discrete_sequence=["#F1EFE8", "#CECBF6", "#AFA9EC", "#7F77DD", "#534AB7", "#26215C"])
fig.update_layout(yaxis_title="Area (km²)", showlegend=False, height=380, margin=dict(t=20))
st.plotly_chart(fig, width="stretch")
st.caption(
    "Each model's binary hazard map uses its own Max-CSI threshold. 0 = no model predicts flood, "
    "5 = all models do. The disagreement zone (1–4) is where the choice of model matters."
)
