"""
RetailSense — Evaluation Module
=================================
Time-series-safe metrics and walk-forward cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Dict, Any


# ── Core Metrics ─────────────────────────────────────────────────────────────

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Percentage Error — the official Rossmann Kaggle metric."""
    mask = y_true > 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — handles near-zero values better than MAPE."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask  = denom > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard MAPE (%)."""
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def coverage_probability(y_true: np.ndarray,
                          y_lower: np.ndarray,
                          y_upper: np.ndarray) -> float:
    """Fraction of actuals inside the prediction interval."""
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))


def evaluate_all(y_true: np.ndarray, y_pred_log: np.ndarray,
                 label: str = 'Model') -> Dict[str, Any]:
    """Compute all metrics. y_pred_log is in log1p space; y_true is raw."""
    y_pred = np.expm1(y_pred_log)
    return {
        'model': label,
        'RMSPE': rmspe(y_true, y_pred),
        'MAE':   mean_absolute_error(y_true, y_pred),
        'RMSE':  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAPE':  mape(y_true, y_pred),
        'sMAPE': smape(y_true, y_pred),
    }


# ── Walk-Forward Cross-Validation ─────────────────────────────────────────────

def walk_forward_cv(model,
                    X: pd.DataFrame,
                    y: pd.Series,
                    y_raw: np.ndarray,
                    n_splits: int = 3) -> List[Dict[str, Any]]:
    """
    Time-series-safe cross-validation using TimeSeriesSplit.
    Returns per-fold metrics. NEVER shuffles data.

    Args:
        model   : sklearn-compatible model with fit/predict
        X       : feature DataFrame (already sorted chronologically)
        y       : log1p(Sales) target
        y_raw   : raw Sales values (for RMSPE)
        n_splits: number of CV folds

    Returns:
        List of metric dicts, one per fold
    """
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va       = X.iloc[va_idx]
        y_true     = y_raw[va_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)

        fold_metrics = evaluate_all(y_true, preds, f'Fold {fold+1}')
        scores.append(fold_metrics)

    return scores


def cv_summary(scores: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean ± std across folds."""
    keys = ['RMSPE', 'MAE', 'RMSE', 'MAPE', 'sMAPE']
    summary = {}
    for k in keys:
        vals = [s[k] for s in scores]
        summary[f'{k}_mean'] = float(np.mean(vals))
        summary[f'{k}_std']  = float(np.std(vals))
    return summary


# ── Inventory Cost ─────────────────────────────────────────────────────────────

def inventory_cost(order: np.ndarray,
                   demand: np.ndarray,
                   holding_cost_per_unit: float,
                   stockout_cost_per_unit: float) -> Dict[str, float]:
    """
    Compute total inventory cost under a given ordering policy.

    Returns dict with total_cost, overstock_cost, understock_cost,
    overstock_units, understock_units.
    """
    overstock  = np.maximum(order - demand, 0)
    understock = np.maximum(demand - order, 0)
    oc = float(np.sum(overstock  * holding_cost_per_unit))
    uc = float(np.sum(understock * stockout_cost_per_unit))
    return {
        'total_cost':       oc + uc,
        'overstock_cost':   oc,
        'understock_cost':  uc,
        'overstock_units':  float(overstock.sum()),
        'understock_units': float(understock.sum()),
        'fill_rate':        float(np.mean(order >= demand)),
    }


def newsvendor_quantile(holding_cost: float, stockout_cost: float) -> float:
    """
    Optimal order quantile from the newsvendor model.
    q* = stockout_cost / (stockout_cost + holding_cost)
    """
    return stockout_cost / (stockout_cost + holding_cost)
