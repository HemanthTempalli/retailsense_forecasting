"""
Page 1 — Store Forecast Dashboard
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Forecast Dashboard | RetailSense", page_icon="📈", layout="wide")

st.markdown("""
<style>
[data-testid="metric-container"]{background:#1e293b;border:1px solid #334155;
border-radius:10px;padding:12px 16px;}
[data-testid="stMetricValue"]{font-size:1.8rem!important;font-weight:700!important;color:#38bdf8!important;}
h1,h2,h3{color:#f1f5f9;}
</style>""", unsafe_allow_html=True)

# ── Load shared artifacts ─────────────────────────────────────────────────────
import json, pickle
from pathlib import Path as P

ARTIFACT_DIR = P(__file__).parent.parent.parent / "artifacts"
@st.cache_resource
def load_store_forecasts():
    demo = not (ARTIFACT_DIR / "store_forecasts.json").exists()
    if not demo:
        with open(ARTIFACT_DIR / "store_forecasts.json") as f:
            return {int(k): v for k, v in json.load(f).items()}, False
    # Demo synthetic data
    np.random.seed(42)
    forecasts = {}
    for sid in range(1, 101):
        base = 4500 + sid * 25
        dates = pd.date_range("2015-06-20", periods=42, freq="D")
        seasonal = 600 * np.sin(np.linspace(0, 4*np.pi, 42))
        promo_boost = np.where(np.arange(42) % 7 < 2, base * 0.18, 0)
        actual = np.maximum(base + seasonal + promo_boost + np.random.normal(0, 120, 42), 500)
        p50 = actual * np.random.uniform(0.93, 1.07, 42)
        std = actual * 0.11
        forecasts[sid] = {
            "dates":  [d.strftime("%Y-%m-%d") for d in dates],
            "actual": actual.tolist(),
            "p10":    np.maximum(p50 - 1.28*std, 0).tolist(),
            "p50":    p50.tolist(),
            "p90":    (p50 + 1.28*std).tolist(),
        }
    return forecasts, True

store_forecasts, demo_mode = load_store_forecasts()
available_stores = sorted(store_forecasts.keys())

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Store Forecast Dashboard")
st.markdown("*42-day demand forecast with 80% prediction intervals per store*")

if demo_mode:
    st.info("⚡ Running in demo mode with synthetic data. Run the Colab notebook and add artifacts to see real model predictions.")

st.divider()

# ── Controls ──────────────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
with col_ctrl1:
    store_id = st.selectbox("🏪 Select Store", available_stores, index=0)
with col_ctrl2:
    show_pi = st.toggle("Show Prediction Interval (p10–p90)", value=True)
with col_ctrl3:
    show_actual = st.toggle("Show Actual Sales", value=True)

# ── Forecast data ─────────────────────────────────────────────────────────────
fc = store_forecasts[store_id]
dates  = pd.to_datetime(fc["dates"])
actual = np.array(fc["actual"])
p10    = np.array(fc["p10"])
p50    = np.array(fc["p50"])
p90    = np.array(fc["p90"])

# ── KPIs ──────────────────────────────────────────────────────────────────────
mae  = np.mean(np.abs(actual - p50))
mape = np.mean(np.abs((actual - p50) / actual)) * 100
cov  = np.mean((actual >= p10) & (actual <= p90))
avg_width = np.mean(p90 - p10)

k1, k2, k3, k4 = st.columns(4)
k1.metric("MAE",           f"€{mae:,.0f}")
k2.metric("MAPE",          f"{mape:.1f}%")
k3.metric("PI Coverage",   f"{cov:.1%}",  delta="target 80%")
k4.metric("Avg PI Width",  f"€{avg_width:,.0f}")

st.divider()

# ── Main forecast chart ───────────────────────────────────────────────────────
fig = go.Figure()

if show_pi:
    # Shaded band
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates, dates[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself',
        fillcolor='rgba(56,189,248,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='80% Prediction Interval',
        hoverinfo='skip',
    ))
    # p90 line
    fig.add_trace(go.Scatter(
        x=dates, y=p90,
        line=dict(color='rgba(56,189,248,0.5)', width=1, dash='dot'),
        name='p90', showlegend=True,
    ))
    # p10 line
    fig.add_trace(go.Scatter(
        x=dates, y=p10,
        line=dict(color='rgba(56,189,248,0.5)', width=1, dash='dot'),
        name='p10', showlegend=True,
    ))

# p50 forecast
fig.add_trace(go.Scatter(
    x=dates, y=p50,
    line=dict(color='#38bdf8', width=2.5),
    name='Forecast (p50)',
    mode='lines+markers',
    marker=dict(size=4),
))

if show_actual:
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        line=dict(color='#4ade80', width=2),
        name='Actual Sales',
        mode='lines+markers',
        marker=dict(size=4),
    ))

fig.update_layout(
    title=f'Store {store_id} — 42-Day Sales Forecast',
    xaxis_title='Date',
    yaxis_title='Daily Sales (€)',
    height=420,
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,23,42,0.6)',
    legend=dict(orientation='h', y=-0.22),
    hovermode='x unified',
)

st.plotly_chart(fig, use_container_width=True)

# ── Day-of-week pattern ───────────────────────────────────────────────────────
st.markdown("### 📅 Demand Pattern by Day of Week")

dow_df = pd.DataFrame({'date': dates, 'p50': p50, 'actual': actual})
dow_df['dow'] = dow_df['date'].dt.day_name()
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_agg = dow_df.groupby('dow')[['p50','actual']].mean().reindex(dow_order)

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=dow_agg.index, y=dow_agg['actual'],
                       name='Avg Actual', marker_color='#4ade80', opacity=0.8))
fig2.add_trace(go.Bar(x=dow_agg.index, y=dow_agg['p50'],
                       name='Avg Forecast', marker_color='#38bdf8', opacity=0.8))
fig2.update_layout(
    height=300, barmode='group', template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
    yaxis_title='Avg Daily Sales (€)',
)
st.plotly_chart(fig2, use_container_width=True)

# ── Forecast table ────────────────────────────────────────────────────────────
st.markdown("### 📋 Forecast Table")
tbl = pd.DataFrame({
    'Date':      dates,
    'Day':       dates.day_name(),
    'Forecast (p50)': [f"€{v:,.0f}" for v in p50],
    'Lower (p10)':    [f"€{v:,.0f}" for v in p10],
    'Upper (p90)':    [f"€{v:,.0f}" for v in p90],
    'Actual':         [f"€{v:,.0f}" for v in actual],
    'Error %':        [f"{abs(a-p)/a*100:.1f}%" for a, p in zip(actual, p50)],
})
st.dataframe(tbl, use_container_width=True, hide_index=True, height=300)
