import plotly.graph_objects as go
import streamlit as st

import data

CENTER = dict(lat=25.49, lon=86.57)

st.title("Maps")
st.caption("Flood susceptibility and hazard maps of Khagaria district on an OpenStreetMap basemap.")

boundary = data.district_boundary()
fig = go.Figure(go.Scattermap())  # an empty trace, so that the map is drawn
fig.update_layout(
    map=dict(style="open-street-map", center=CENTER, zoom=9.3,
             layers=[dict(sourcetype="geojson", source=boundary, type="line",
                          color="#2C2C2A", line=dict(width=2))]),
    height=620, margin=dict(l=0, r=0, t=0, b=0),
)
st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
