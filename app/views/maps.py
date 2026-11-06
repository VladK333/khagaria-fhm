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
AGREEMENT = {f"{n} model{'s' * (n != 1)}": c for n, c in
             enumerate(["#F1EFE8", "#CECBF6", "#AFA9EC", "#7F77DD", "#534AB7", "#26215C"])}
LAYERS = {"Probability": "probability", "Risk classes": "classes", "Binary hazard": "binary",
          "Errors": "errors", "Model agreement": "agreement"}
LEGENDS = {"Risk classes": CLASSES, "Binary hazard": BINARY, "Errors": ERRORS, "Model agreement": AGREEMENT}
ALL_MODELS = {"Model agreement"}  # layers that combine all models instead of showing one
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


def map_figure(layer: str, model: str, opacity: float, height: int) -> go.Figure:
    """The selected layer of one model over the basemap, with the district boundary and a legend."""
    image, corners = data.map_layer(LAYERS[layer] if layer in ALL_MODELS else f"{LAYERS[layer]}_{model}")
    fig = go.Figure(colour_bar() if layer == "Probability" else class_legend(LEGENDS[layer]))
    fig.update_layout(
        map=dict(style="open-street-map", center=CENTER, zoom=9.3 if height > 500 else 8.8,
                 layers=[dict(sourcetype="image", source=image, coordinates=corners, opacity=opacity),
                         dict(sourcetype="geojson", source=data.district_boundary(), type="line",
                              color="#2C2C2A", line=dict(width=2))]),
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        # Lifted above the OpenStreetMap attribution, which spans the bottom of narrow maps
        legend=dict(x=0.01, y=0.01 if height > 500 else 0.07, yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.85)", font=dict(color="#2C2C2A", size=11)),
    )
    return fig


def layer_metrics(layer: str, model: str) -> list[tuple[str, str]]:
    """Key numbers shown under the map, as (label, value) pairs."""
    if layer == "Binary hazard":
        area = data.area_statistics().set_index("Model").loc[model]
        return [("Max-CSI threshold", f"{area['MaxCSI_threshold']:.3f}"),
                ("Flood hazard area", f"{area['Hazard_km2']:,.0f} km²"),
                ("Share of district", f"{area['Hazard_pct']:.1f} %")]
    if layer == "Errors":
        c = data.error_counts_district().loc[model]
        test_csi = data.test_metrics().query("Threshold_method == 'MaxCSI'").set_index("Model").loc[model, "CSI"]
        return [("False alarms (FP)", f"{c['FP'] * PIXEL_KM2:,.0f} km²"),
                ("Missed floods (FN)", f"{c['FN'] * PIXEL_KM2:,.0f} km²"),
                ("CSI, whole district", f"{c['TP'] / (c['TP'] + c['FP'] + c['FN']):.3f}"),
                ("CSI, test split", f"{test_csi:.3f}")]
    if layer == "Model agreement":
        agree = data.agreement_areas().set_index("models_flagging")["km2"]
        return [("All models agree", f"{100 * (agree[0] + agree[5]) / agree.sum():.1f} %"),
                ("Flood in all 5 models", f"{agree[5]:,.0f} km²"),
                ("Models disagree", f"{agree[1:5].sum():,.0f} km²")]
    return []


def show_metrics(metrics: list[tuple[str, str]], per_row: int) -> None:
    for start in range(0, len(metrics), per_row):
        for col, (label, value) in zip(st.columns(per_row), metrics[start:start + per_row]):
            col.metric(label, value)


CAPTIONS = {
    "Probability": "Probability of flooding predicted by the model for every 30 m pixel (shown at 60 m). "
                   "Outside the district the map is transparent.",
    "Risk classes": "Probabilities split into five classes with the natural breaks of the XGBoost map ({breaks}), "
                    "the same breaks for every model. Area per class is on the Area statistics page.",
    "Binary hazard": "Pixels where the probability is at or above the model's Max-CSI threshold, which was selected "
                     "on the validation split. For comparison, the SAR flood inventory covers 824 km² (56.1 %).",
    "Errors": "The hazard map compared with the SAR flood inventory, pixel by pixel. This map covers every district "
              "pixel, including the 70 % the model was trained on, so it looks better than the model really is for "
              "models that overfit: compare the two CSI values. The honest score is the one on the test split.",
    "Model agreement": "How many of the five models flag each pixel as flood, each at its own Max-CSI threshold. "
                       "The light purple fringe (1–4 models) is where the choice of model changes the hazard map.",
}

st.title("Maps")
st.caption("Flood susceptibility and hazard maps of Khagaria district on an OpenStreetMap basemap.")

layer = st.segmented_control("Layer", list(LAYERS), default="Probability")
if layer is None:
    layer = "Probability"
compare = st.toggle("Compare two models", disabled=layer in ALL_MODELS) and layer not in ALL_MODELS
opacity = st.slider("Opacity", 0.0, 1.0, 0.8, step=0.05)

if compare:
    sides = st.columns(2)
    for side, default in zip(sides, ["XGBoost", "ANN"]):
        with side:
            model = st.selectbox("Model", data.MODELS, index=data.MODELS.index(default), key=f"model_{default}")
            st.plotly_chart(map_figure(layer, model, opacity, height=500), width="stretch",
                            config={"scrollZoom": True}, key=f"map_{default}")
            show_metrics(layer_metrics(layer, model), per_row=2)
else:
    model = st.selectbox("Model", data.MODELS, disabled=layer in ALL_MODELS)
    st.plotly_chart(map_figure(layer, model, opacity, height=620), width="stretch", config={"scrollZoom": True})
    metrics = layer_metrics(layer, model)
    if metrics:
        show_metrics(metrics, per_row=len(metrics))

breaks = ", ".join(f"{b:.3f}" for b in data.run_config()["natural_breaks"])
st.caption(CAPTIONS[layer].format(breaks=breaks))
