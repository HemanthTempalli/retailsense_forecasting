"""
RetailSense — Demand Forecasting & Inventory Intelligence
=========================================================
Multi-page Streamlit app for the Rossmann forecasting pipeline.

Pages:
  1. Overview Dashboard
  2. Store Forecast
  3. Inventory Optimizer
  4. Model Explainability
  5. What-If Simulator
"""

import streamlit as st

st.set_page_config(
    page_title="RetailSense | Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {background: #0f172a;}
    [data-testid="stSidebar"] * {color: #e2e8f0 !important;}

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Headers */
    h1, h2, h3 {color: #f1f5f9;}

    /* KPI numbers */
    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }

    /* Dividers */
    hr {border-color: #334155;}

    /* Badge */
    .badge {
        display: inline-block;
        background: #1d4ed8;
        color: #dbeafe;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-green  {background:#065f46; color:#d1fae5;}
    .badge-orange {background:#92400e; color:#fef3c7;}
    .badge-red    {background:#7f1d1d; color:#fee2e2;}

    /* Card */
    .info-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


# ── Artifact loader (cached) ──────────────────────────────────────────────────
ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts"
@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    """Load all models and metadata from artifacts/."""
    art = {}
    demo_mode = not (ARTIFACT_DIR / "lgbm_tuned.pkl").exists()
    art['demo_mode'] = demo_mode

    if not demo_mode:
        with open(ARTIFACT_DIR / "lgbm_tuned.pkl", "rb") as f:
            art['lgbm'] = pickle.load(f)
        with open(ARTIFACT_DIR / "xgb_model.pkl", "rb") as f:
            art['xgb'] = pickle.load(f)
        with open(ARTIFACT_DIR / "meta_model.pkl", "rb") as f:
            art['meta'] = pickle.load(f)
        with open(ARTIFACT_DIR / "quantile_models.pkl", "rb") as f:
            art['quantiles'] = pickle.load(f)
        with open(ARTIFACT_DIR / "shap_explainer.pkl", "rb") as f:
            art['shap_explainer'] = None
        if (ARTIFACT_DIR / "shap_values_sample.parquet").exists():
            art['shap_values'] = pd.read_parquet(ARTIFACT_DIR / "shap_values_sample.parquet").values
            art['shap_X']      = pd.read_parquet(ARTIFACT_DIR / "shap_X_sample.parquet")
        if (ARTIFACT_DIR / "store_meta.csv").exists():
            art['store_meta']  = pd.read_csv(ARTIFACT_DIR / "store_meta.csv")

    if (ARTIFACT_DIR / "metadata.json").exists():
        with open(ARTIFACT_DIR / "metadata.json") as f:
            art['metadata'] = json.load(f)
    else:
        art['metadata'] = _demo_metadata()

    if (ARTIFACT_DIR / "store_forecasts.json").exists():
        with open(ARTIFACT_DIR / "store_forecasts.json") as f:
            art['store_forecasts'] = {int(k): v for k, v in json.load(f).items()}
    else:
        art['store_forecasts'] = _demo_forecasts()

    if (ARTIFACT_DIR / "cost_params.json").exists():
        with open(ARTIFACT_DIR / "cost_params.json") as f:
            art['cost_params'] = json.load(f)
    else:
        art['cost_params'] = {"unit_cost": 8, "selling_price": 15,
                               "holding_cost_rate": 0.20, "stockout_penalty": 5,
                               "critical_ratio": 0.98, "savings_pct": 17.3}
    return art


def _demo_metadata():
    return {
        "best_model": "Stacking Ensemble (final)",
        "rmspe": 0.1124, "mae": 612,
        "n_stores": 1115, "pi_coverage": 0.812,
        "model_results": [
            {"model": "Stacking Ensemble (final)", "RMSPE": 0.1124, "MAE": 612},
            {"model": "LightGBM (Optuna tuned)",   "RMSPE": 0.1187, "MAE": 638},
            {"model": "XGBoost (baseline)",         "RMSPE": 0.1341, "MAE": 724},
            {"model": "Prophet (top-10 stores)",    "RMSPE": 0.1602, "MAE": 891},
        ],
        "feature_cols": [],
        "val_start": "2015-06-20", "val_end": "2015-07-31",
    }


def _demo_forecasts():
    """Synthetic demo forecasts for Streamlit Cloud (no artifacts)."""
    np.random.seed(42)
    forecasts = {}
    for store_id in range(1, 51):
        base = 5000 + store_id * 30
        dates = pd.date_range("2015-06-20", periods=42, freq="D")
        trend = np.linspace(0, 200, 42)
        seasonal = 500 * np.sin(np.linspace(0, 4 * np.pi, 42))
        noise = np.random.normal(0, 150, 42)
        actual = np.maximum(base + trend + seasonal + noise, 500)
        p50    = actual * np.random.uniform(0.92, 1.08, 42)
        std    = actual * 0.12
        forecasts[store_id] = {
            "dates":  [d.strftime("%Y-%m-%d") for d in dates],
            "actual": actual.tolist(),
            "p10":    (p50 - 1.28 * std).tolist(),
            "p50":    p50.tolist(),
            "p90":    (p50 + 1.28 * std).tolist(),
        }
    return forecasts


# ── Shared state ──────────────────────────────────────────────────────────────
if "artifacts" not in st.session_state:
    st.session_state.artifacts = load_artifacts()

art      = st.session_state.artifacts
metadata = art["metadata"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 RetailSense")
    st.markdown("**Demand Forecasting & Inventory Intelligence**")
    st.divider()

    if art["demo_mode"]:
        st.warning("⚡ **Demo Mode**\nRun the Colab notebook to load real models.", icon="⚠️")
    else:
        st.success("✅ Models loaded", icon="✅")

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("""
- ✅ Rossmann Store Sales dataset
- ✅ 38 engineered features
- ✅ Walk-forward CV (no leakage)
- ✅ XGBoost · LightGBM · Prophet
- ✅ Optuna hyperparameter tuning
- ✅ Stacking ensemble
- ✅ Quantile regression (p10/p50/p90)
- ✅ SHAP explainability
- ✅ Newsvendor inventory optimizer
    """)

    st.divider()
    st.markdown(f"""
<div style='font-size:0.8rem; color:#94a3b8;'>
Best model: <b>{metadata['best_model']}</b><br>
RMSPE: <b>{metadata['rmspe']:.4f}</b><br>
MAE: <b>€{metadata['mae']:,.0f}</b><br>
PI Coverage: <b>{metadata['pi_coverage']:.1%}</b><br>
Stores: <b>{metadata['n_stores']:,}</b>
</div>
""", unsafe_allow_html=True)

    st.divider()
    


# ── Home Page ─────────────────────────────────────────────────────────────────
st.markdown("# 🛒 RetailSense — Demand Forecasting Platform")
st.markdown("*End-to-end ML forecasting for 1,115 retail stores · 42-day horizon · Uncertainty-aware inventory optimization*")
st.divider()

# KPI row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Best RMSPE",    f"{metadata['rmspe']:.4f}", delta="↓ vs baseline")
col2.metric("MAE",           f"€{metadata['mae']:,.0f}")
col3.metric("PI Coverage",   f"{metadata['pi_coverage']:.1%}", delta="target 80%")
col4.metric("Stores",        f"{metadata['n_stores']:,}")
col5.metric("Forecast Days", "42")

st.divider()

# Navigation cards
st.markdown("### 🗺️ Navigate to a module:")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown("""<div class='info-card'>
    <h4>📈 Forecast Dashboard</h4>
    <p style='font-size:0.85rem;color:#94a3b8;'>42-day sales forecast per store with 80% prediction intervals</p>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class='info-card'>
    <h4>📦 Inventory Optimizer</h4>
    <p style='font-size:0.85rem;color:#94a3b8;'>Newsvendor-optimal order quantities minimising total cost</p>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown("""<div class='info-card'>
    <h4>🔍 Explainability</h4>
    <p style='font-size:0.85rem;color:#94a3b8;'>SHAP feature importance + per-prediction waterfall explanations</p>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown("""<div class='info-card'>
    <h4>🔬 What-If Simulator</h4>
    <p style='font-size:0.85rem;color:#94a3b8;'>Toggle promotions, holidays — see forecast update instantly</p>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown("""<div class='info-card'>
    <h4>🏆 Model Comparison</h4>
    <p style='font-size:0.85rem;color:#94a3b8;'>XGBoost vs LightGBM vs Prophet vs Stacking Ensemble</p>
    </div>""", unsafe_allow_html=True)

st.divider()

# Model leaderboard
st.markdown("### 🏆 Model Leaderboard")
results_df = pd.DataFrame(metadata["model_results"])
results_df["RMSPE_bar"] = results_df["RMSPE"]

st.dataframe(
    results_df.style
        .format({"RMSPE": "{:.4f}", "MAE": "€{:,.0f}"})
        .background_gradient(subset=["RMSPE"], cmap="RdYlGn_r")
        .set_properties(**{"font-size": "14px"}),
    use_container_width=True,
    hide_index=True,
)

# Architecture
st.divider()
st.markdown("### 🏗️ ML Pipeline Architecture")
st.markdown("""
```
Raw Data (train.csv + store.csv)
         │
         ▼
   Data Cleaning ──── Remove closed stores, impute, encode
         │
         ▼
Feature Engineering ─ Lag(7,14,28,56d) · Rolling(7,14,28d) · Fourier · Calendar
         │
         ▼
Walk-Forward Split ── Train: Jan 2013–Jun 2015  │  Val: Jun–Jul 2015 (42d)
         │
         ├──────────────────────────────────────────────────────────┐
         ▼                                                          ▼
   XGBoost (300 trees)                          LightGBM (Optuna: 25 trials)
         │                                                          │
         └────────────────── Stacking (Ridge) ─────────────────────┘
                                    │
                                    ▼
                     Quantile Models (p10 · p50 · p90)
                                    │
                                    ▼
                      SHAP Explainability · Newsvendor Inventory
```
""")

st.caption("Use the sidebar pages ↑ to explore each module.")
