"""Interactive dashboard. Run with: streamlit run src/anomaly_detector/app.py"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from anomaly_detector.cli import build
from anomaly_detector.data import make_series
from anomaly_detector.metrics import evaluate, segments, threshold_from_train

st.set_page_config(page_title="anomaly-detector", page_icon="📈", layout="wide")
st.title("anomaly-detector")
st.caption("Time-series anomaly detection · built by [Abbishekk DGS](https://abbishekkdgs.tech)")

with st.sidebar:
    source = st.radio("Data", ["Synthetic (labelled)", "Upload CSV"])
    if source == "Upload CSV":
        upload = st.file_uploader("CSV", type="csv")
        column = st.text_input("Value column", "value")
        period = st.number_input("Seasonal period (rows, 0 = none)", 0, 10_000, 0)
    else:
        seed = st.number_input("Seed", 0, 100, 0)
        period = 288
    train_frac = st.slider("Leading fraction known to be normal", 0.1, 0.6, 0.3)


@st.cache_data
def run(values: tuple, period: int, train_frac: float):
    import numpy as np

    x = np.array(values)
    split = int(len(x) * train_frac)
    fit_end = int(split * 2 / 3)
    ens = build(period or None)
    ens.fit(x[:fit_end], calibration=x[fit_end:split])
    calib = ens.score_all(x[fit_end:split])
    scores = ens.score_all(x)
    thr = threshold_from_train(calib["ensemble"])
    return scores, thr, split


labels = None
if source == "Upload CSV":
    if not upload:
        st.info("Upload a CSV with a numeric column to begin.")
        st.stop()
    df = pd.read_csv(upload)
    x = df[column].astype(float).interpolate().to_numpy()
    index = df.index
else:
    s = make_series(seed=int(seed))
    x, labels, index = s.values, s.labels, s.timestamps

with st.spinner("Fitting detectors..."):
    scores, thr, split = run(tuple(x), int(period), train_frac)
flags = scores["ensemble"] >= thr

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.04
)
fig.add_trace(go.Scatter(x=index, y=x, name="value", line=dict(width=1)), row=1, col=1)
if labels is not None:
    for s0, e0 in segments(labels):
        fig.add_vrect(
            x0=index[s0],
            x1=index[min(e0, len(index) - 1)],
            fillcolor="orange",
            opacity=0.2,
            line_width=0,
            row=1,
            col=1,
        )
fig.add_trace(
    go.Scatter(
        x=index[flags], y=x[flags], mode="markers", name="flagged", marker=dict(color="red", size=5)
    ),
    row=1,
    col=1,
)
for name in ("robust_z", "iforest", "lstm_ae", "ensemble"):
    fig.add_trace(go.Scatter(x=index, y=scores[name], name=name, line=dict(width=1)), row=2, col=1)
fig.add_hline(y=thr, line_dash="dash", row=2, col=1)
fig.add_vline(x=index[split], line_dash="dot")
fig.update_layout(height=650, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, width="stretch")
st.caption(
    "Orange bands: true anomalies · red dots: flagged · dotted line: end of normal reference data"
)

if labels is not None:
    m = evaluate(scores["ensemble"][split:], labels[split:], thr)
    c = st.columns(4)
    c[0].metric("Point-adjusted F1", f"{m['pa_f1']:.2f}")
    c[1].metric("Events detected", m["events_detected"])
    c[2].metric("False-alarm episodes", m["false_alarm_episodes"])
    c[3].metric("Raw F1", f"{m['f1']:.2f}")
