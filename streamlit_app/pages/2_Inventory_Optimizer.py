"""
Page 2 — Inventory Optimizer
Newsvendor model + cost curve visualisation
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Inventory Optimizer | RetailSense", page_icon="📦", layout="wide")

st.markdown("""
<style>
[data-testid="metric-container"]{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 16px;}
[data-testid="stMetricValue"]{font-size:1.8rem!important;font-weight:700!important;color:#38bdf8!important;}
h1,h2,h3{color:#f1f5f9;}
.result-card{background:#0f172a;border:1px solid #1d4ed8;border-radius:12px;padding:20px 24px;margin:10px 0;}
</style>""", unsafe_allow_html=True)

from scipy.stats import norm

ARTIFACT_DIR = Path(__file__).parent.parent.parent / "artifacts"


def load_cost_params():
    path = ARTIFACT_DIR / "cost_params.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"unit_cost": 8.0, "selling_price": 15.0, "holding_cost_rate": 0.20,
            "stockout_penalty": 5.0, "critical_ratio": 0.98, "savings_pct": 17.3}


def load_store_forecasts():
    path = ARTIFACT_DIR / "store_forecasts.json"
    if path.exists():
        with open(path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    # Demo
    np.random.seed(42)
    forecasts = {}
    for sid in range(1, 51):
        base = 4800 + sid * 30
        dates = pd.date_range("2015-06-20", periods=42, freq="D")
        actual = np.maximum(base + np.random.normal(0, 400, 42), 500)
        p50 = actual * np.random.uniform(0.93, 1.07, 42)
        std = actual * 0.11
        forecasts[sid] = {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "actual": actual.tolist(),
            "p10": np.maximum(p50 - 1.28*std, 0).tolist(),
            "p50": p50.tolist(),
            "p90": (p50 + 1.28*std).tolist(),
        }
    return forecasts


cost_defaults = load_cost_params()
store_forecasts = load_store_forecasts()
available_stores = sorted(store_forecasts.keys())

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📦 Inventory Optimizer")
st.markdown("*Newsvendor-optimal order quantities that minimise total expected inventory cost*")
st.divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Cost Parameters")

unit_cost    = st.sidebar.number_input("Unit Cost (€)",         value=float(cost_defaults["unit_cost"]),    min_value=0.1, step=0.5)
sell_price   = st.sidebar.number_input("Selling Price (€)",     value=float(cost_defaults["selling_price"]), min_value=0.1, step=0.5)
hold_rate    = st.sidebar.slider("Annual Holding Cost Rate",     min_value=0.05, max_value=0.50,
                                  value=float(cost_defaults["holding_cost_rate"]), step=0.01, format="%0.0f%%")
stockout_pen = st.sidebar.number_input("Stockout Penalty (€/unit)", value=float(cost_defaults["stockout_penalty"]), min_value=0.0, step=0.5)
store_id     = st.sidebar.selectbox("Select Store", available_stores)

# ── Derived ───────────────────────────────────────────────────────────────────
holding_day  = unit_cost * hold_rate / 365
stockout_c   = (sell_price - unit_cost) + stockout_pen
critical_ratio = stockout_c / (stockout_c + holding_day)
z_star = norm.ppf(critical_ratio)

# Forecasts for selected store
fc = store_forecasts[store_id]
p50 = np.array(fc["p50"])
p10 = np.array(fc["p10"])
p90 = np.array(fc["p90"])

# Approx std from PI
approx_std = (np.array(fc["p90"]) - np.array(fc["p10"])) / (2 * 1.28)

# Optimal order per day
optimal_orders = np.maximum(p50 + z_star * approx_std, 0)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Optimal Service Level", f"{critical_ratio:.1%}")
k2.metric("Holding Cost / unit / day", f"€{holding_day:.4f}")
k3.metric("Stockout Cost / unit", f"€{stockout_c:.2f}")
k4.metric("Newsvendor z-score", f"{z_star:.2f}")

st.divider()

# ── Cost curve ────────────────────────────────────────────────────────────────
st.markdown("### 📉 Expected Cost Curve (Day 1 Example)")

day_idx = 0
mu  = p50[day_idx]
sig = max(approx_std[day_idx], 1)

q_range = np.linspace(max(0, mu - 3*sig), mu + 3*sig, 100)
holding_costs = []
stockout_costs = []

for q in q_range:
    z = (q - mu) / sig
    exp_over  = max(0, sig * (norm.pdf(z) - z * (1 - norm.cdf(z))))
    exp_under = max(0, sig * (norm.pdf(z) + z * norm.cdf(z) - z))
    holding_costs.append(exp_over  * holding_day)
    stockout_costs.append(exp_under * stockout_c)

total_costs = np.array(holding_costs) + np.array(stockout_costs)
opt_q = q_range[np.argmin(total_costs)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=q_range, y=total_costs,
    name='Total Cost', line=dict(color='#f472b6', width=3)))
fig.add_trace(go.Scatter(x=q_range, y=holding_costs,
    name='Holding Cost', line=dict(color='#fb923c', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=q_range, y=stockout_costs,
    name='Stockout Cost', line=dict(color='#f87171', width=2, dash='dash')))
fig.add_vline(x=opt_q, line=dict(color='#4ade80', width=2, dash='dot'),
              annotation_text=f"Optimal: {opt_q:,.0f}", annotation_position="top right")
fig.add_vline(x=mu, line=dict(color='#38bdf8', width=1.5, dash='dot'),
              annotation_text=f"p50: {mu:,.0f}", annotation_position="top left")

fig.update_layout(
    title='Expected Cost vs Order Quantity (Newsvendor Model)',
    xaxis_title='Order Quantity (units)', yaxis_title='Expected Cost (€)',
    height=380, template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
    legend=dict(orientation='h', y=-0.25),
)
st.plotly_chart(fig, use_container_width=True)

# ── 6-week order plan ─────────────────────────────────────────────────────────
st.markdown("### 📅 6-Week Optimal Order Plan")

dates = pd.to_datetime(fc["dates"])
week_data = []
for w in range(6):
    sl = slice(w*7, (w+1)*7)
    wk_p50  = p50[sl]
    wk_std  = approx_std[sl]
    wk_opt  = optimal_orders[sl]
    wk_act  = np.array(fc["actual"])[sl]
    week_data.append({
        "Week":           f"Week {w+1}",
        "Start":          dates[w*7].strftime("%b %d"),
        "Forecast Total": f"€{wk_p50.sum():,.0f}",
        "Optimal Order":  f"{wk_opt.sum():,.0f} units",
        "Safety Stock":   f"{max(0, (wk_opt - wk_p50).sum()):,.0f} units",
        "Actual":         f"€{wk_act.sum():,.0f}",
    })

st.dataframe(pd.DataFrame(week_data), use_container_width=True, hide_index=True)

# ── Daily order chart ─────────────────────────────────────────────────────────
st.markdown("### 📦 Daily Forecast vs Optimal Order")

fig3 = go.Figure()
fig3.add_trace(go.Bar(x=dates, y=p50, name='Forecast (p50)',
                       marker_color='#38bdf8', opacity=0.7))
fig3.add_trace(go.Scatter(x=dates, y=optimal_orders, name='Optimal Order',
                           line=dict(color='#4ade80', width=2.5),
                           mode='lines+markers', marker=dict(size=5)))
fig3.add_trace(go.Scatter(
    x=np.concatenate([dates, dates[::-1]]),
    y=np.concatenate([p90, p10[::-1]]),
    fill='toself', fillcolor='rgba(56,189,248,0.1)',
    line=dict(color='rgba(0,0,0,0)'), name='80% PI', hoverinfo='skip',
))

fig3.update_layout(
    height=350, template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
    xaxis_title='Date', yaxis_title='Units / Sales (€)',
    legend=dict(orientation='h', y=-0.25),
    title=f'Store {store_id} — Optimal Order Quantities',
)
st.plotly_chart(fig3, use_container_width=True)

# ── Cost saving summary ───────────────────────────────────────────────────────
st.divider()
st.markdown("### 💰 Business Impact Summary")

naive_order = p50 * 0.95
actual_arr  = np.array(fc["actual"])

def total_cost(order, demand):
    over  = np.maximum(order - demand, 0)
    under = np.maximum(demand - order, 0)
    return float(np.sum(over * holding_day + under * stockout_c))

naive_cost = total_cost(naive_order, actual_arr)
opt_cost   = total_cost(optimal_orders, actual_arr)
savings    = naive_cost - opt_cost
savings_pct = savings / naive_cost * 100 if naive_cost > 0 else 0

b1, b2, b3 = st.columns(3)
b1.metric("Naive Ordering Cost",   f"€{naive_cost:,.0f}")
b2.metric("Optimised Cost",        f"€{opt_cost:,.0f}", delta=f"-€{savings:,.0f}")
b3.metric("Cost Reduction",        f"{savings_pct:.1f}%", delta="over 42 days")
