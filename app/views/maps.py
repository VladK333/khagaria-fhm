import plotly.graph_objects as go
import streamlit as st

import data

CENTER = dict(lat=25.49, lon=86.57)
# Same colours as scripts/viz/prepare_map_layers.py
CLASSES = {"Very Low": "#E1F5EE", "Low": "#9FE1CB", "Moderate": "#FAC775", "High": "#F0997B",
           "Very High": "#D85A30"}
BINARY = {"Flood": "#185FA5", "Non-flood": "#D3D1C7"}
ERRORS = {"Hit (TP)": "#1D9E75", "False alarm (FP)": "#EF9F27", "Missed (FN)": "#E24B4A",
          "Correct non-flood (TN)": "#D3D1C7"}
LAYERS = {"Probability": "probability", "Risk classes": "classes", "Binary hazard": "binary",
          "Errors": "errors"}
LEGENDS = {"Risk classes": CLASSES, "Binary hazard": BINARY, "Errors": ERRORS}
PIXEL_KM2 = 0.03 * 0.03


def colour_bar() -> list[go.Scattermap]:
    """An invisible trace that only draws a colour bar for the probability image."""
    return [go.Scattermap(lat=[None], lon=[None], mode="markers", hoverinfo="skip", showlegend=False,
                          marker=dict(color=[0], colorscale="Blues", cmin=0, cmax=1, showscale=True,
                                      colorbar=dict(title="Flood<br>probability", thickness=14)))]


def class_legend(classes: dict[str, str]) -> list[go.Scattermap]:
    """Invisible traces that only add one legend entry per class colour."""
    return [go.Scattermap(lat=[None], lon=[None], mode="markers", name=name, hoverinfo="skip",
                          marker=dict(size=14, color=colour)) for name, colour in classes.items()]


st.title("Maps")
st.caption("Flood susceptibility and hazard maps of Khagaria district on an OpenStreetMap basemap.")

layer = st.segmented_control("Layer", list(LAYERS), default="Probability")
if layer is None:
    layer = "Probability"
cols = st.columns(2)
model = cols[0].selectbox("Model", data.MODELS)
opacity = cols[1].slider("Opacity", 0.0, 1.0, 0.8, step=0.05)

image, corners = data.map_layer(f"{LAYERS[layer]}_{model}")
fig = go.Figure(colour_bar() if layer == "Probability" else class_legend(LEGENDS[layer]))
fig.update_layout(
    map=dict(style="open-street-map", center=CENTER, zoom=9.3,
             layers=[dict(sourcetype="image", source=image, coordinates=corners, opacity=opacity),
                     dict(sourcetype="geojson", source=data.district_boundary(), type="line",
                          color="#2C2C2A", line=dict(width=2))]),
    height=620, margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.85)", font=dict(color="#2C2C2A")),
)
st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})

if layer == "Probability":
    st.caption(f"Probability of flooding predicted by {model} for every 30 m pixel (shown at 60 m). "
               "Outside the district the map is transparent.")
elif layer == "Binary hazard":
    area = data.area_statistics().set_index("Model").loc[model]
    m = st.columns(3)
    m[0].metric("Max-CSI threshold", f"{area['MaxCSI_threshold']:.3f}")
    m[1].metric("Flood hazard area", f"{area['Hazard_km2']:,.0f} km²")
    m[2].metric("Share of district", f"{area['Hazard_pct']:.1f} %")
    st.caption(f"Pixels where the {model} probability is at or above its Max-CSI threshold, which was selected "
               "on the validation split. For comparison, the SAR flood inventory covers 824 km² (56.1 %).")
elif layer == "Errors":
    c = data.error_counts_district().loc[model]
    test_csi = data.test_metrics().query("Threshold_method == 'MaxCSI'").set_index("Model").loc[model, "CSI"]
    m = st.columns(4)
    m[0].metric("False alarms (FP)", f"{c['FP'] * PIXEL_KM2:,.0f} km²")
    m[1].metric("Missed floods (FN)", f"{c['FN'] * PIXEL_KM2:,.0f} km²")
    m[2].metric("CSI, whole district", f"{c['TP'] / (c['TP'] + c['FP'] + c['FN']):.3f}")
    m[3].metric("CSI, test split", f"{test_csi:.3f}")
    st.caption(
        f"The {model} hazard map compared with the SAR flood inventory, pixel by pixel. This map covers every "
        "district pixel, including the 70 % the model was trained on, so it looks better than the model really is "
        "for models that overfit: compare the two CSI values. The honest score is the one on the test split."
    )
else:
    breaks = data.run_config()["natural_breaks"]
    st.caption(f"{model} probabilities split into five classes with the natural breaks of the XGBoost map "
               f"({', '.join(f'{b:.3f}' for b in breaks)}), the same breaks for every model. "
               "Area per class is on the Area statistics page.")
