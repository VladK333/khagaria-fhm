"""Shared map drawing for the Maps and Input factors pages."""
import plotly.graph_objects as go

import data

CENTER = dict(lat=25.49, lon=86.57)


def colour_bar(colorscale, cmin: float, cmax: float, title: str) -> list[go.Scattermap]:
    """An invisible trace that only draws a colour bar for an image layer."""
    return [go.Scattermap(lat=[None], lon=[None], mode="markers", hoverinfo="skip", showlegend=False,
                          marker=dict(color=[cmin], colorscale=colorscale, cmin=cmin, cmax=cmax, showscale=True,
                                      colorbar=dict(title=title, thickness=14)))]


def class_legend(classes: dict[str, str]) -> list[go.Scattermap]:
    """Invisible traces that only add one legend entry per class colour."""
    return [go.Scattermap(lat=[None], lon=[None], mode="markers", name=name, hoverinfo="skip",
                          marker=dict(size=14, color=colour)) for name, colour in classes.items()]


def map_figure(layer: str, legend: list[go.Scattermap], opacity: float, height: int) -> go.Figure:
    """A pre-rendered layer (see data.map_layer) over the basemap, with the district boundary and a legend."""
    image, corners = data.map_layer(layer)
    fig = go.Figure(legend)
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
