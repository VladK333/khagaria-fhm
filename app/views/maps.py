import plotly.graph_objects as go
import streamlit as st

import data

CENTER = dict(lat=25.49, lon=86.57)

st.title("Maps")
st.caption("Flood susceptibility and hazard maps of Khagaria district on an OpenStreetMap basemap.")

cols = st.columns(2)
model = cols[0].selectbox("Model", data.MODELS)
opacity = cols[1].slider("Opacity", 0.0, 1.0, 0.8, step=0.05)

image, corners = data.map_layer(f"probability_{model}")
# An invisible trace that only draws the colour bar for the image layer
legend = go.Scattermap(lat=[None], lon=[None], mode="markers", hoverinfo="skip", showlegend=False,
                       marker=dict(color=[0], colorscale="Blues", cmin=0, cmax=1, showscale=True,
                                   colorbar=dict(title="Flood<br>probability", thickness=14)))
fig = go.Figure(legend)
fig.update_layout(
    map=dict(style="open-street-map", center=CENTER, zoom=9.3,
             layers=[dict(sourcetype="image", source=image, coordinates=corners, opacity=opacity),
                     dict(sourcetype="geojson", source=data.district_boundary(), type="line",
                          color="#2C2C2A", line=dict(width=2))]),
    height=620, margin=dict(l=0, r=0, t=0, b=0),
)
st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
st.caption(
    f"Probability of flooding predicted by {model} for every 30 m pixel (shown at 60 m). "
    "Outside the district the map is transparent."
)
