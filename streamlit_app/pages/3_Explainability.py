"""
Page 3 — Model Explainability (SHAP)
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Explainability | RetailSense", page_icon="🔍", layout="wide")

st.markdown("""
<style>
[data-testid="metric-container"]{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 16px;}
[data-testid="stMetricValue"]{font-size:1.8rem!important;font-weight:700!important;color:#38bdf8!important;}
h1,h2,h3{color:#f1f5f9;}
</style>""", unsafe_allow_html=True)

ARTIFACT_DIR = Path(__file__).parent.parent.parent / "artifacts"


@st.cache_resource
def load_shap_data():
    sv_path = ARTIFACT_DIR / "shap_values_sample.parquet"
    sx_path = ARTIFACT_DIR / "shap_X_sample.parquet"
    if sv_path.exists() and sx_path.exists():
        shap_values = pd.read_parquet(sv_path).values
        shap_X      = pd.read_parquet(sx_path)
        return shap_values, shap_X, False
    # Demo synthetic
    feature_cols = [
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
    np.random.seed(42)
    n = 500
    shap_X = pd.DataFrame(np.random.randn(n, len(feature_cols)), columns=feature_cols)
    # Realistic SHAP importance distribution
    importance = np.array([
        0.3, 0.8, 1.2, 0.2, 0.4, 0.5, 0.3, 0.6, 0.4,
        0.2, 0.3, 0.5, 0.7, 0.4, 0.6, 0.3, 0.9, 0.2, 0.3,
        0.8, 0.7, 0.6, 0.5, 0.4, 0.5,
        2.5, 2.1, 1.8, 1.5,
        1.9, 0.8, 1.6, 1.4, 0.7, 1.3, 1.0, 0.6, 1.2,
        1.7
    ])
    shap_values = np.random.randn(n, len(feature_cols)) * importance
    return shap_values, shap_X, True


shap_values, shap_X, demo_mode = load_shap_data()
feature_cols = list(shap_X.columns)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Model Explainability")
st.markdown("*SHAP (SHapley Additive exPlanations) — understanding why the model makes each prediction*")

if demo_mode:
    st.info("⚡ Demo mode: synthetic SHAP values shown. Run Colab notebook to see real model explanations.")

st.divider()

# ── Global feature importance ──────────────────────────────────────────────────
st.markdown("### 🌍 Global Feature Importance (Mean |SHAP|)")

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
mean_abs_shap = mean_abs_shap.sort_values(ascending=True).tail(20)

color_map = []
for feat in mean_abs_shap.index:
    if 'lag' in feat or 'roll' in feat or 'expanding' in feat:
        color_map.append('#38bdf8')  # blue = lag/rolling
    elif 'sin' in feat or 'cos' in feat or feat in ['Month','DayOfWeek','WeekOfYear','Quarter']:
        color_map.append('#a78bfa')  # purple = seasonality
    elif feat in ['Promo','Promo2','Promo2Active']:
        color_map.append('#4ade80')  # green = promo
    else:
        color_map.append('#fb923c')  # orange = other

fig_imp = go.Figure(go.Bar(
    x=mean_abs_shap.values,
    y=mean_abs_shap.index,
    orientation='h',
    marker_color=color_map,
    text=[f'{v:.4f}' for v in mean_abs_shap.values],
    textposition='outside',
))

fig_imp.update_layout(
    height=550, template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
    xaxis_title='Mean |SHAP value|',
    title='Top-20 Features by SHAP Importance',
)
st.plotly_chart(fig_imp, use_container_width=True)

# Legend
col_l1, col_l2, col_l3, col_l4 = st.columns(4)
col_l1.markdown("🔵 **Lag / Rolling features**")
col_l2.markdown("🟣 **Seasonality / Calendar**")
col_l3.markdown("🟢 **Promotion features**")
col_l4.markdown("🟠 **Store / competition**")

st.divider()

# ── Beeswarm scatter ───────────────────────────────────────────────────────────
st.markdown("### 🐝 SHAP Beeswarm — Feature Value Impact")
st.markdown("*Each dot = one prediction. Colour = feature value (red=high, blue=low).*")

top_n = st.slider("Show top N features", min_value=5, max_value=20, value=12)

top_features = mean_abs_shap.sort_values(ascending=False).head(top_n).index.tolist()

fig_bee = go.Figure()

for i, feat in enumerate(reversed(top_features)):
    feat_idx = feature_cols.index(feat)
    sv = shap_values[:, feat_idx]
    fv = shap_X[feat].values
    fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)

    fig_bee.add_trace(go.Scatter(
        x=sv,
        y=np.full_like(sv, i) + np.random.uniform(-0.3, 0.3, len(sv)),
        mode='markers',
        marker=dict(
            size=4,
            color=fv_norm,
            colorscale='RdBu_r',
            opacity=0.6,
        ),
        name=feat,
        showlegend=False,
        hovertemplate=f'<b>{feat}</b><br>SHAP: %{{x:.4f}}<br>Value: %{{text}}<extra></extra>',
        text=[f'{v:.2f}' for v in fv],
    ))

fig_bee.update_layout(
    height=520,
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,23,42,0.6)',
    xaxis_title='SHAP Value (impact on log-sales)',
    yaxis=dict(tickvals=list(range(top_n)),
               ticktext=list(reversed(top_features))),
    title='SHAP Beeswarm Plot',
    shapes=[dict(type='line', x0=0, x1=0, y0=-0.5, y1=top_n-0.5,
                 line=dict(color='#475569', width=1.5, dash='dot'))],
)
st.plotly_chart(fig_bee, use_container_width=True)

# ── Waterfall — single prediction ─────────────────────────────────────────────
st.divider()
st.markdown("### 💧 Waterfall Plot — Single Prediction Explanation")
st.markdown("*Why did the model predict this particular sales value for this day?*")

pred_idx = st.slider("Select prediction index", 0, min(len(shap_values)-1, 100), 0)

sv_row = shap_values[pred_idx]
base_value = np.log1p(5000)  # approximate

# Top contributing features
top_k = 12
top_idx = np.argsort(np.abs(sv_row))[::-1][:top_k]
top_feats  = [feature_cols[i] for i in top_idx]
top_shap   = [sv_row[i] for i in top_idx]
top_values = [shap_X.iloc[pred_idx][feature_cols[i]] for i in top_idx]

# Build waterfall
running = base_value
x_vals, y_labels, measure, text_vals, colors = [], [], [], [], []

y_labels.append('Base Value')
x_vals.append(base_value)
measure.append('absolute')
text_vals.append(f'{base_value:.3f}')
colors.append('#94a3b8')

for feat, sv_val, fv in zip(reversed(top_feats), reversed(top_shap), reversed(top_values)):
    y_labels.append(f'{feat}={fv:.2f}')
    x_vals.append(sv_val)
    measure.append('relative')
    text_vals.append(f'{sv_val:+.4f}')
    colors.append('#4ade80' if sv_val > 0 else '#f87171')

y_labels.append('Prediction')
x_vals.append(0)
measure.append('total')
text_vals.append(f'{base_value + sum(top_shap):.3f}')
colors.append('#38bdf8')

fig_wf = go.Figure(go.Waterfall(
    orientation='h',
    x=x_vals,
    y=y_labels,
    measure=measure,
    text=text_vals,
    textposition='outside',
    connector=dict(line=dict(color='#475569', width=1)),
    increasing=dict(marker_color='#4ade80'),
    decreasing=dict(marker_color='#f87171'),
    totals=dict(marker_color='#38bdf8'),
))

fig_wf.update_layout(
    height=480, template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
    title=f'SHAP Waterfall — Prediction #{pred_idx}',
    xaxis_title='log(1+Sales) contribution',
    showlegend=False,
)
st.plotly_chart(fig_wf, use_container_width=True)

# Summary
pred_log = base_value + sum(top_shap)
pred_sales = np.expm1(pred_log)
st.info(f"📊 This prediction: **€{pred_sales:,.0f}** daily sales "
        f"(log-scale: {pred_log:.3f})")
