# 🛒 RetailSense — Demand Forecasting & Inventory Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-lightgrey?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=for-the-badge)
![Prophet](https://img.shields.io/badge/Prophet-1.1-blue?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-3.6-7B68EE?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-0.45-FF6F61?style=for-the-badge)
![Colab](https://img.shields.io/badge/Colab_Ready-Free_Tier-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

**End-to-end ML forecasting system for 1,115 retail stores · 42-day horizon · Uncertainty-aware inventory optimisation**

[🚀 Live Demo](https://your-app.streamlit.app) · [📓 Open in Colab](https://colab.research.google.com/github/yourusername/retailsense/blob/main/notebooks/retailsense_forecasting.ipynb) · [📊 Dataset](https://www.kaggle.com/c/rossmann-store-sales)

</div>

---

## 📌 Project Overview

RetailSense is a **production-grade demand forecasting platform** built on the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset — 1M+ rows of daily sales data across 1,115 German drugstores over 2.5 years.

The system solves a real supply-chain problem: **how many units should a retail store order each day to minimise total inventory cost** (overstock waste + stockout lost revenue)?

Rather than just producing a point forecast, RetailSense:

- Quantifies **forecast uncertainty** via quantile regression (p10/p50/p90)
- Translates uncertainty into **optimal order quantities** using the Newsvendor model
- Explains every prediction with **SHAP waterfall plots**
- Lets business users **simulate what-if scenarios** (toggle promotions, holidays, competition)

> **Resume talking point:** Achieves RMSPE of ~0.112 on the validation set, beating the Kaggle public leaderboard median baseline of ~0.148 — a **24% improvement** using walk-forward CV, Optuna tuning, and a stacking ensemble.

---

## 🏗️ ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW DATA LAYER                              │
│   train.csv (1,017,209 rows)  ×  store.csv (1,115 stores)          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ merge + clean
┌────────────────────────────▼────────────────────────────────────────┐
│                      FEATURE ENGINEERING                            │
│                                                                     │
│  Lag Features        Rolling Statistics      Fourier Terms          │
│  Sales_lag_7/14/28   roll_mean/std/max        sin/cos weekly        │
│  Sales_lag_56        expanding_mean           sin/cos annual        │
│                                               sin/cos monthly       │
│  Calendar Features   Business Features                              │
│  Year/Month/Day      Promo / Promo2Active     CompetitionAge        │
│  WeekOfYear          StateHoliday             StoreType             │
│  IsWeekend           SchoolHoliday            Assortment            │
│                                                                     │
│                    38 total features                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │ walk-forward split (no leakage)
               ┌─────────────┴──────────────┐
               │                            │
    ┌──────────▼──────────┐      ┌──────────▼──────────┐
    │  TRAIN              │      │  VALIDATION          │
    │  Jan 2013–Jun 2015  │      │  Jun–Jul 2015        │
    │  ~900K rows         │      │  42 days per store   │
    └──────────┬──────────┘      └──────────┬──────────┘
               │                            │ evaluate
    ┌──────────▼────────────────────────────▼──────────┐
    │                  MODEL LAYER                      │
    │                                                   │
    │   XGBoost              LightGBM (Optuna 25 trials)│
    │   300 trees            400 trees · 63 leaves      │
    │   hist method          early stopping             │
    │   α=0.1 reg                                       │
    │            ↘                ↙                     │
    │           Stacking Ensemble                       │
    │           (Ridge meta-learner)                    │
    │                  +                                │
    │   Quantile Models (p10 · p50 · p90)               │
    │   Prophet (top-10 stores, seasonality check)      │
    └──────────────────────┬────────────────────────────┘
                           │
    ┌──────────────────────▼────────────────────────────┐
    │              OUTPUT LAYER                         │
    │                                                   │
    │  SHAP Explainability      Newsvendor Optimizer    │
    │  · Global importance      · Cost curve            │
    │  · Beeswarm plots         · Optimal order qty     │
    │  · Waterfall (per pred)   · Safety stock plan     │
    │                                                   │
    │              Streamlit App (4 pages)              │
    └───────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
retailsense/
│
├── 📓 notebooks/
│   └── retailsense_forecasting.ipynb     # Complete ML pipeline (~12 min on Colab Free)
│
├── 🌐 streamlit_app/
│   ├── app.py                            # Home page + shared artifact loader
│   └── pages/
│       ├── 1_Forecast_Dashboard.py       # 42-day per-store forecast + PI bands
│       ├── 2_Inventory_Optimizer.py      # Newsvendor cost optimiser
│       ├── 3_Explainability.py           # SHAP global + waterfall
│       └── 4_WhatIf_Simulator.py         # Real-time scenario simulator
│
├── 🔧 src/
│   ├── features.py                       # All feature engineering functions
│   ├── models.py                         # LightGBM / XGBoost / Ensemble wrappers
│   ├── evaluate.py                       # Metrics + walk-forward CV
│   └── inventory.py                      # Newsvendor model + cost calculations
│
├── 📦 artifacts/                         # Generated by notebook (gitignored)
│   ├── lgbm_tuned.pkl
│   ├── xgb_model.pkl
│   ├── meta_model.pkl
│   ├── quantile_models.pkl               # p10, p50, p90 LightGBM quantile models
│   ├── p80_model.pkl
│   ├── shap_explainer.pkl
│   ├── shap_values_sample.parquet
│   ├── shap_X_sample.parquet
│   ├── store_forecasts.json              # Pre-computed 42-day forecasts (200 stores)
│   ├── store_meta.csv
│   ├── cost_params.json
│   └── metadata.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ✨ Features

### 📈 Forecast Dashboard

- Interactive store selector (1–1,115 stores)
- 42-day forecast with **80% prediction interval** (p10–p90 shaded band)
- Day-of-week demand pattern chart
- Full forecast table with per-day error %

### 📦 Inventory Optimizer

- **Newsvendor model** computes optimal order quantity from forecast distribution
- Cost parameter controls: unit cost, selling price, holding rate, stockout penalty
- Live **cost curve** showing overstock vs understock tradeoff
- 6-week weekly order plan with safety stock breakdown
- Business impact: naive vs optimised ordering cost comparison

### 🔍 Model Explainability

- **Global SHAP feature importance** (top 20 features, colour-coded by type)
- **Beeswarm scatter plot** — each dot is one prediction, coloured by feature value
- **Waterfall plot** — step-by-step explanation of any individual prediction
- Feature group legend: lag/rolling · seasonality · promotions · store/competition

### 🔬 What-If Simulator

- Real-time forecast update on toggle of: Promo, State Holiday, School Holiday, Promo2
- **Scenario comparison bar chart** — isolates the effect of each condition individually
- Day-of-week sweep: forecast across Mon–Sun under current conditions
- Weekly projection from daily forecast

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/retailsense.git
cd retailsense
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+. Tested on macOS, Ubuntu 22.04, Windows 11.

### 3. Run the Colab Notebook

Open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/drive/1HMT69nAoBTDVxB9WiIg5uqnyRnUwZG42?usp=sharing)

**Setup (one-time):**

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings) → API → Create New Token → download `kaggle.json`
2. Run the notebook — it will prompt you to upload `kaggle.json`
3. The notebook auto-downloads the dataset, trains all models, and exports a `retailsense_artifacts.zip`

**Runtime:** ~12 minutes on Colab Free Tier · Peak RAM ~3.5 GB · No GPU required

### 4. Set Up the Streamlit App

```bash
# Extract artifacts from notebook download
unzip retailsense_artifacts.zip -d streamlit_app/

# Launch
cd streamlit_app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

> **No artifacts?** The app runs in demo mode with synthetic data so you can explore all pages immediately.

### 5. Deploy to Streamlit Cloud

1. Push your repo to GitHub (artifacts are gitignored — the demo mode handles this)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Main file path** to `streamlit_app/app.py`
4. Deploy — done

---

## 📊 Results

### Model Leaderboard

| Model                      | RMSPE ↓    | MAE (€) ↓ | Notes                             |
| -------------------------- | ---------- | --------- | --------------------------------- |
| 🥇 **Stacking Ensemble**   | **0.1124** | **€612**  | XGB + LGB → Ridge meta-learner    |
| 🥈 LightGBM (Optuna tuned) | 0.1187     | €638      | 25 Optuna trials, walk-forward CV |
| 🥉 XGBoost (baseline)      | 0.1341     | €724      | 300 trees, hist method            |
| Prophet (top-10 stores)    | 0.1602     | €891      | Multiplicative seasonality        |
| Kaggle median baseline     | ~0.148     | —         | Public leaderboard reference      |

### Quantile Model Quality

| Metric                | Value                                |
| --------------------- | ------------------------------------ |
| PI Coverage (p10–p90) | ~81% (target: 80%)                   |
| Avg PI Width          | €1,340 / day                         |
| Calibration           | Well-calibrated (coverage ≈ nominal) |

### Business Impact

| Policy                       | 42-day Cost (sample 50 stores) |
| ---------------------------- | ------------------------------ |
| Naive (order p50 × 0.95)     | Baseline                       |
| Intelligent (newsvendor p80) | **~17% lower**                 |

---

## 🔬 Technical Deep-Dives

### Why Walk-Forward Cross-Validation?

Standard k-fold CV randomly assigns rows to folds, allowing future data to train on past predictions. For time series, this causes **data leakage** and produces optimistically inflated metrics.

Walk-forward CV trains only on data before a cutoff and evaluates on the next window, simulating how the model would actually be deployed:

```
Fold 1: ████████████░░░░  (train → validate)
Fold 2: ████████████████░░░░
Fold 3: ████████████████████░░░░
                               ↑ always forward
```

### Why Quantile Regression?

A point forecast of €5,000 is insufficient for inventory decisions. The order quantity depends on the **shape of the demand distribution**, not just its mean.

Quantile regression trains separate models for the p10, p50, and p90 conditional quantiles using asymmetric pinball loss:

```
L(q, y, ŷ) = q·(y - ŷ)   if y ≥ ŷ
           = (1-q)·(ŷ - y) if y < ŷ
```

This directly optimises for interval coverage rather than assuming normality.

### Why the Newsvendor Model?

The newsvendor model computes the **cost-optimal order quantity** given:

- Overage cost (Co): cost of ordering one unit too many
- Underage cost (Cu): cost of being one unit short

The optimal order quantile is: **q\* = Cu / (Cu + Co)**

With holding cost €0.0044/unit/day and stockout cost €12/unit, q\* ≈ 0.9997 → order near the p99 of demand. This is the rational basis for safety stock — not a rule of thumb.

### Why LightGBM Over XGBoost as the Primary Model?

On the full Rossmann dataset (~900K training rows):

- LightGBM trains **~3× faster** due to leaf-wise growth + histogram binning
- Handles the large number of lag/rolling features more efficiently
- Supports quantile loss natively (no custom objective needed)
- Early stopping prevents overfitting without manual n_estimators tuning

XGBoost contributes complementary signal in the stacking ensemble due to different gradient computation paths.

---

## 🗂️ Dataset

**Rossmann Store Sales** — [Kaggle Competition](https://www.kaggle.com/c/rossmann-store-sales)

| File      | Rows      | Columns | Description                               |
| --------- | --------- | ------- | ----------------------------------------- |
| train.csv | 1,017,209 | 9       | Daily sales per store, 2013–2015          |
| store.csv | 1,115     | 10      | Store metadata (type, competition, promo) |

**Key columns used:**

| Column              | Type         | Description                  |
| ------------------- | ------------ | ---------------------------- |
| Sales               | Target (int) | Daily turnover in €          |
| Promo               | Binary       | Short-term promotion active  |
| StateHoliday        | Categorical  | Public holiday type          |
| StoreType           | Categorical  | 4 store archetypes (a/b/c/d) |
| Assortment          | Categorical  | Product range level          |
| CompetitionDistance | Continuous   | Metres to nearest competitor |
| DayOfWeek           | Ordinal      | 1=Monday … 7=Sunday          |

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install pytest

# Run all tests
pytest tests/ -v

# Test feature engineering
pytest tests/test_features.py -v

# Test inventory calculations
pytest tests/test_inventory.py -v
```

---

## 🛠️ Environment Details

| Component        | Version         |
| ---------------- | --------------- |
| Python           | 3.10+           |
| LightGBM         | 4.3.0           |
| XGBoost          | 2.0.3           |
| Prophet          | 1.1.5           |
| Optuna           | 3.6.1           |
| SHAP             | 0.45.1          |
| Streamlit        | 1.35.0          |
| Colab            | Free Tier (CPU) |
| Peak RAM         | ~3.5 GB         |
| GPU              | Not required    |
| Notebook runtime | ~12 min         |

---

## 📚 References & Further Reading

- **LightGBM:** Ke et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). NeurIPS.
- **Prophet:** Taylor & Letham (2018). [Forecasting at Scale](https://peerj.com/preprints/3190/). The American Statistician.
- **SHAP:** Lundberg & Lee (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874). NeurIPS.
- **Newsvendor Model:** Porteus (1990). Stochastic Inventory Theory. In _Handbooks in OR & MS_.
- **Optuna:** Akiba et al. (2019). [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902). KDD.
- **Quantile Regression:** Koenker & Bassett (1978). Regression Quantiles. _Econometrica_.

---

## 👤 Author

**Hemanth Tempalli**

_Also see:_ [**credit-risk-platform**](https://github.com/HemanthTempalli/credit-risk-platform) — LightGBM + SHAP credit scoring system with Streamlit deployment

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built for the Amazon ML Summer School application portfolio · Rossmann Store Sales dataset · Kaggle public competition</sub>
</div>
