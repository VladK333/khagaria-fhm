import plotly.graph_objects as go
import streamlit as st

import data

st.title("Threshold explorer")
st.caption(
    "A model outputs a flood probability per pixel; the threshold τ turns it into flood / non-flood. "
    "Max-CSI and Youden thresholds were selected on the validation split; the curves are on the test split."
)

model = st.selectbox("Model", data.MODELS)
curve = data.threshold_curve(model)
marks = data.run_config()["thresholds"][model]

key = f"tau_{model}"
if key not in st.session_state:
    st.session_state[key] = round(marks["MaxCSI"], 3)


def snap(value: float) -> None:
    st.session_state[key] = round(value, 3)


cols = st.columns([4, 1, 1])
tau = cols[0].slider("Threshold τ", 0.0, 1.0, step=0.001, format="%.3f", key=key)
cols[1].button(f"Max-CSI ({marks['MaxCSI']:.3f})", on_click=snap, args=(marks["MaxCSI"],),
               use_container_width=True)
cols[2].button(f"Youden ({marks['Youden']:.3f})", on_click=snap, args=(marks["Youden"],),
               use_container_width=True)

row = curve.iloc[(curve["threshold"] - tau).abs().idxmin()]
best = curve.loc[curve["CSI"].idxmax()]

m = st.columns(4)
m[0].metric("CSI", f"{row['CSI']:.3f}",
            help=f"Highest CSI on the test set: {best['CSI']:.3f} at τ = {best['threshold']:.3f}")
m[1].metric("Sensitivity", f"{row['Sensitivity']:.3f}")
m[2].metric("Specificity", f"{row['Specificity']:.3f}")
m[3].metric("Precision", f"{row['Precision']:.3f}")

fig = go.Figure()
for name, color in [("CSI", "#534AB7"), ("Sensitivity", "#1D9E75"), ("Specificity", "#D85A30")]:
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve[name], name=name,
                             line=dict(color=color, width=2)))
fig.add_vline(x=tau, line_dash="dash", line_color="gray")
for label, value, side in [("Max-CSI", marks["MaxCSI"], "top left"),
                           ("Youden", marks["Youden"], "top right")]:
    fig.add_vline(x=value, line_dash="dot", line_color="rgba(128,128,128,0.6)",
                  annotation_text=label, annotation_position=side)
fig.update_layout(xaxis_title="Threshold τ", yaxis_title="Score", yaxis_range=[0, 1.05],
                  hovermode="x unified", margin=dict(t=30), legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Confusion counts on the test set")
c = st.columns(4)
c[0].metric("True positives", f"{int(row['TP']):,}", help="Flooded pixels predicted as flood")
c[1].metric("False negatives", f"{int(row['FN']):,}", help="Flooded pixels missed by the model")
c[2].metric("False positives", f"{int(row['FP']):,}", help="Dry pixels predicted as flood")
c[3].metric("True negatives", f"{int(row['TN']):,}", help="Dry pixels predicted as dry")
