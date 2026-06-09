"""
Page 4 — What-If Simulator
Toggle promotions, holidays, store features → live forecast update
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="What-If Simulator | RetailSense", page_icon="🔬", layout="wide")

st.markdown("""
<style>
[data-testid="metric-container"]{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 16px;}
[data-testid="stMetricValue"]{font-size:1.9rem!important;font-weight:700!important;color:#38bdf8!important;}
[data-testid="stMetricDelta"]{font-size:1rem!important;}
h1,h2,h3{color:#f1f5f9;}
.scenario-card{background:#0f172a;border:1px solid #334155;border-radius:12px;padding:16px 20px;margin:8px 0;}
</style>""", unsafe_allow_html=True)

ARTIFACT_DIR = Path(__file__).parent.parent.parent / "artifacts"


@st.cache_resource
def load_model():
    path = ARTIFACT_DIR / "lgbm_tuned.pkl"

    try:
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f), False
    except Exception as e:
        st.warning(f"Model load failed: {e}")

    return None, True

model, demo_mode = load_model()

FEATURE_COLS = [
    'Store','DayOfWeek','Promo','StateHoliday','SchoolHoliday',
    'StoreType','Assortment','CompetitionDistance','CompetitionAge',
    'Promo2','Promo2Active','Year','Month','Day','WeekOfYear','Quarter',
    'IsWeekend','IsMonthStart','IsMonthEnd',
    'sin_week','cos_week','sin_year','cos_year','sin_month','cos_month',
    'Sales_lag_7','Sales_lag_14','Sales_lag_28','Sales_lag_56',
    'Sales_roll_mean_7','Sales_roll_std_7','Sales_roll_max_7',
    'Sales_roll_mean_14','Sales_roll_std_14','Sales_roll_max_14',
    'Sales_roll_mean_28','Sales_roll_std_28','Sales_roll_max_28',
    'Sales_expanding_mean'
]


def demo_predict(features: dict) -> float:
    """Analytical approximation when model not loaded."""
    base = features.get('Sales_expanding_mean', 5000)
    if base == 0:
        base = 5000
    multiplier = 1.0
    if features.get('Promo', 0) == 1:
        multiplier *= 1.18
    if features.get('StateHoliday', 0) != 0:
        multiplier *= 0.70
    if features.get('SchoolHoliday', 0) == 1:
        multiplier *= 0.88
    dow = features.get('DayOfWeek', 3)
    dow_mults = {1: 1.05, 2: 1.0, 3: 1.0, 4: 1.02, 5: 1.08, 6: 0.72, 7: 0.0}
    multiplier *= dow_mults.get(dow, 1.0)
    if features.get('CompetitionDistance', 1000) < 500:
        multiplier *= 0.94
    return base * multiplier


def predict_sales(features: dict) -> float:
    if model is None:
        return demo_predict(features)
    row = pd.DataFrame([features])[FEATURE_COLS]
    return float(np.expm1(model.predict(row)[0]))


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🔬 What-If Simulator")
st.markdown("*Change business conditions and see how the model forecasts change in real time*")

if demo_mode:
    st.info("⚡ Demo mode: using analytical approximation. Load real model artifacts for true ML predictions.")

st.divider()

# ── Base scenario ──────────────────────────────────────────────────────────────
st.markdown("### 🏪 Store & Date Settings")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    store_id    = st.number_input("Store ID", 1, 1115, 1)
    store_type  = st.selectbox("Store Type", [0, 1, 2, 3],
                                format_func=lambda x: ['A','B','C','D'][x])
    assortment  = st.selectbox("Assortment", [0, 1, 2],
                                format_func=lambda x: ['Basic','Extra','Extended'][x])

with col_s2:
    dow           = st.selectbox("Day of Week",
                                  [1,2,3,4,5,6,7],
                                  format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x-1])
    month         = st.selectbox("Month", list(range(1,13)),
                                  format_func=lambda x: pd.Timestamp(2015, x, 1).strftime('%B'))
    year          = st.selectbox("Year", [2013, 2014, 2015], index=2)

with col_s3:
    comp_dist = st.slider("Competition Distance (m)", 100, 20000, 2000, 100)
    lag_7     = st.number_input("Last Week Sales (€)", 1000, 30000, 5500, 100)
    expanding = st.number_input("Store Avg Sales (€)", 1000, 30000, 5200, 100)


# ── Business toggles ───────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🎚️ Toggle Business Conditions")

col_t1, col_t2, col_t3, col_t4 = st.columns(4)
with col_t1:
    promo = st.toggle("🏷️ Promo Active", value=False)
with col_t2:
    state_holiday = st.toggle("🎌 State Holiday", value=False)
with col_t3:
    school_holiday = st.toggle("🏫 School Holiday", value=False)
with col_t4:
    promo2 = st.toggle("🔖 Promo2 Running", value=False)

# ── Build feature dict ─────────────────────────────────────────────────────────
day_of_year = pd.Timestamp(year, month, 1).dayofyear
week_of_year = pd.Timestamp(year, month, 1).isocalendar().week

base_features = {
    'Store': store_id,
    'DayOfWeek': dow,
    'Promo': int(promo),
    'StateHoliday': int(state_holiday),
    'SchoolHoliday': int(school_holiday),
    'StoreType': store_type,
    'Assortment': assortment,
    'CompetitionDistance': comp_dist,
    'CompetitionAge': max(0, 12 * (year - 2012) + (month - 6)),
    'Promo2': int(promo2),
    'Promo2Active': int(promo2),
    'Year': year,
    'Month': month,
    'Day': 15,
    'WeekOfYear': int(week_of_year),
    'Quarter': (month - 1) // 3 + 1,
    'IsWeekend': int(dow >= 6),
    'IsMonthStart': 0,
    'IsMonthEnd': 0,
    'sin_week': np.sin(2 * np.pi * dow / 7),
    'cos_week': np.cos(2 * np.pi * dow / 7),
    'sin_year': np.sin(2 * np.pi * day_of_year / 365),
    'cos_year': np.cos(2 * np.pi * day_of_year / 365),
    'sin_month': np.sin(2 * np.pi * month / 12),
    'cos_month': np.cos(2 * np.pi * month / 12),
    'Sales_lag_7': lag_7,
    'Sales_lag_14': lag_7 * 0.97,
    'Sales_lag_28': lag_7 * 0.95,
    'Sales_lag_56': lag_7 * 0.93,
    'Sales_roll_mean_7': lag_7,
    'Sales_roll_std_7': lag_7 * 0.08,
    'Sales_roll_max_7': lag_7 * 1.15,
    'Sales_roll_mean_14': lag_7 * 0.98,
    'Sales_roll_std_14': lag_7 * 0.09,
    'Sales_roll_max_14': lag_7 * 1.20,
    'Sales_roll_mean_28': lag_7 * 0.97,
    'Sales_roll_std_28': lag_7 * 0.10,
    'Sales_roll_max_28': lag_7 * 1.25,
    'Sales_expanding_mean': expanding,
}

# Baseline (no promo, no holiday)
baseline_features = base_features.copy()
baseline_features['Promo'] = 0
baseline_features['StateHoliday'] = 0
baseline_features['SchoolHoliday'] = 0
baseline_features['Promo2'] = 0
baseline_features['Promo2Active'] = 0

forecast_val  = predict_sales(base_features)
baseline_val  = predict_sales(baseline_features)
delta_val     = forecast_val - baseline_val
delta_pct     = delta_val / baseline_val * 100 if baseline_val > 0 else 0

# ── Results ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📊 Forecast Results")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Forecast (current scenario)", f"€{forecast_val:,.0f}")
r2.metric("Baseline (no promotions)",    f"€{baseline_val:,.0f}")
r3.metric("Δ vs Baseline",              f"€{delta_val:+,.0f}",
          delta=f"{delta_pct:+.1f}%",
          delta_color="normal" if delta_val >= 0 else "inverse")
r4.metric("Weekly Projection",          f"€{forecast_val * 6:,.0f}")

# ── Scenario comparison bar chart ─────────────────────────────────────────────
st.divider()
st.markdown("### 🔀 Scenario Comparison")

scenarios = {"Baseline": baseline_val}

# Add individual effect scenarios
promo_feat = {**baseline_features, 'Promo': 1}
holiday_feat = {**baseline_features, 'StateHoliday': 1}
school_feat  = {**baseline_features, 'SchoolHoliday': 1}
promo2_feat  = {**baseline_features, 'Promo2': 1, 'Promo2Active': 1}
all_feat     = base_features

scenarios["+ Promo only"]         = predict_sales(promo_feat)
scenarios["+ State Holiday only"]  = predict_sales(holiday_feat)
scenarios["+ School Holiday only"] = predict_sales(school_feat)
scenarios["+ Promo2 only"]        = predict_sales(promo2_feat)
scenarios["Current (combined)"]   = forecast_val

colors = []
for k, v in scenarios.items():
    if v > baseline_val * 1.05:
        colors.append('#4ade80')
    elif v < baseline_val * 0.95:
        colors.append('#f87171')
    else:
        colors.append('#94a3b8')
colors[0] = '#38bdf8'   # baseline always blue

fig = go.Figure(go.Bar(
    x=list(scenarios.keys()),
    y=list(scenarios.values()),
    marker_color=colors,
    text=[f'€{v:,.0f}' for v in scenarios.values()],
    textposition='outside',
))
fig.add_hline(y=baseline_val, line=dict(color='#38bdf8', dash='dot', width=1.5),
              annotation_text='Baseline')

fig.update_layout(
    height=380, template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
    yaxis_title='Forecast Daily Sales (€)',
    title='Impact of Each Business Condition',
    yaxis=dict(range=[min(scenarios.values())*0.85, max(scenarios.values())*1.12]),
)
st.plotly_chart(fig, use_container_width=True)

# ── Day-of-week sweep ──────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📅 Forecast Across All Days (Same Conditions)")

dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
dow_vals  = []
for d in range(1, 8):
    f = {**base_features, 'DayOfWeek': d,
         'IsWeekend': int(d >= 6),
         'sin_week': np.sin(2*np.pi*d/7),
         'cos_week': np.cos(2*np.pi*d/7)}
    dow_vals.append(predict_sales(f))

fig2 = go.Figure(go.Bar(
    x=dow_names, y=dow_vals,
    marker_color=['#f87171' if d >= 5 else '#38bdf8' for d in range(7)],
    text=[f'€{v:,.0f}' for v in dow_vals],
    textposition='outside',
))
fig2.update_layout(
    height=320, template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
    yaxis_title='Forecast Sales (€)',
    title='Weekly Pattern (current conditions, day varied)',
)
st.plotly_chart(fig2, use_container_width=True)
