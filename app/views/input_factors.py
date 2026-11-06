import streamlit as st

import data
from mapping import class_legend, colour_bar, map_figure
from widgets import pick

DESCRIPTIONS = {
    "Aspect": ("°", "Direction the slope faces, clockwise from north; −1 marks flat cells."),
    "Curvature": ("", "Surface curvature: negative is concave (collects water), positive is convex."),
    "Dis2Roads": ("m", "Distance to the nearest road."),
    "Dis2stream": ("m", "Distance to the nearest stream."),
    "Elevation": ("m", "Height above sea level. The district is a flat floodplain, so the range is only ~12 m."),
    "Geomorphology": ("", "Landform class."),
    "Lithology": ("", "Rock and sediment type."),
    "LULC": ("", "Land use / land cover class."),
    "Rainfall": ("mm", "Rainfall, interpolated over the district."),
    "Slope": ("°", "Terrain slope, recomputed in v2 from Elevation (the submitted version used a copy of TWI)."),
    "Soil": ("", "Soil unit. The straight east–west boundary comes from the ~1 km source soil map."),
    "SPI": ("", "Stream power index: erosive power of flowing water."),
    "Stream_Density": ("", "Length of streams per unit area."),
    "TWI": ("", "Topographic wetness index: where water tends to accumulate."),
}

st.title("Input factors")
st.caption("The 14 conditioning factors every model was trained on, on the same basemap as the result maps.")

features = data.run_config()["features"]
categorical = set(data.run_config()["categorical_coded"])
factor = pick("Factor", features, "Rainfall", key="factor", pills=True)
opacity = st.slider("Opacity", 0.0, 1.0, 0.85, step=0.05)

unit, description = DESCRIPTIONS[factor]
legend = data.map_legend(f"factor_{factor}")
if factor in categorical:
    traces = class_legend({f"Class {code}": colour for code, colour in legend["classes"].items()})
else:
    title = f"{factor}<br>({unit})" if unit else factor
    traces = colour_bar(legend["colorscale"], legend["vmin"], legend["vmax"], title)
st.plotly_chart(map_figure(f"factor_{factor}", traces, opacity, height=620), width="stretch",
                config={"scrollZoom": True})

ranks = data.mean_abs_shap().query("Feature == @factor").set_index("Model")["rank"]
m = st.columns(3)
m[0].metric("Type", "Categorical" if factor in categorical else "Continuous")
m[1].metric("SHAP rank (mean of 5 models)", f"{ranks.mean():.1f} of 14")
m[2].metric("VIF", f"{data.vif().set_index('Feature').loc[factor, 'VIF']:.2f}")

note = ("Class names are not in the repository yet, so classes are shown by code."
        if factor in categorical else "Colours are stretched between the 2nd and 98th percentile.")
st.caption(f"{description} {note}")
