"""
RetailSense — Inventory Optimization Module
=============================================
Newsvendor model + cost-optimal safety stock calculations.
"""

import numpy as np
from typing import Dict, Tuple


def newsvendor_order(forecast_p50: float,
                     forecast_std: float,
                     unit_cost: float,
                     selling_price: float,
                     holding_cost_rate: float = 0.20,
                     stockout_penalty: float = 5.0,
                     days_per_year: float = 365.0) -> Dict[str, float]:
    """
    Compute optimal order quantity using the Newsvendor model.

    The newsvendor model finds the order quantity that minimises
    expected total cost = overstock_cost + understock_cost.

    Args:
        forecast_p50      : median (point) forecast
        forecast_std      : standard deviation of forecast (uncertainty)
        unit_cost         : cost to purchase/produce one unit
        selling_price     : revenue per unit sold
        holding_cost_rate : annual holding cost as fraction of unit cost
        stockout_penalty  : additional € penalty per unfulfilled unit
        days_per_year     : days denominator for daily holding cost

    Returns:
        dict with optimal_order, safety_stock, service_level, costs
    """
    holding_cost_day  = unit_cost * holding_cost_rate / days_per_year
    stockout_cost     = (selling_price - unit_cost) + stockout_penalty

    # Critical ratio = optimal service level
    critical_ratio    = stockout_cost / (stockout_cost + holding_cost_day)

    # Optimal order = µ + z * σ   (normal demand assumption)
    from scipy.stats import norm
    z_optimal         = norm.ppf(critical_ratio)
    optimal_order     = max(0, forecast_p50 + z_optimal * forecast_std)
    safety_stock      = max(0, z_optimal * forecast_std)

    # Expected costs
    exp_overstock  = forecast_std * (norm.pdf(z_optimal) - z_optimal * (1 - critical_ratio))
    exp_understock = forecast_std * (norm.pdf(z_optimal) + z_optimal * critical_ratio - z_optimal)
    exp_overstock  = max(0, exp_overstock)
    exp_understock = max(0, exp_understock)

    return {
        'optimal_order':    round(optimal_order, 1),
        'safety_stock':     round(safety_stock, 1),
        'service_level':    round(critical_ratio * 100, 1),
        'critical_ratio':   round(critical_ratio, 4),
        'z_score':          round(z_optimal, 3),
        'holding_cost_day': round(holding_cost_day, 4),
        'stockout_cost':    round(stockout_cost, 2),
        'exp_holding_cost': round(exp_overstock * holding_cost_day, 2),
        'exp_stockout_cost':round(exp_understock * stockout_cost, 2),
        'exp_total_cost':   round(exp_overstock * holding_cost_day +
                                  exp_understock * stockout_cost, 2),
    }


def cost_curve(forecast_p50: float,
               forecast_std: float,
               unit_cost: float,
               selling_price: float,
               holding_cost_rate: float = 0.20,
               stockout_penalty: float = 5.0,
               n_points: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute total cost curve over a range of order quantities.
    Returns (quantities, total_costs, holding_costs, stockout_costs)
    """
    from scipy.stats import norm

    holding_cost_day = unit_cost * holding_cost_rate / 365
    stockout_cost    = (selling_price - unit_cost) + stockout_penalty

    # Demand range
    lo = max(0, forecast_p50 - 3 * forecast_std)
    hi = forecast_p50 + 3 * forecast_std
    quantities = np.linspace(lo, hi, n_points)

    total_costs    = []
    holding_costs  = []
    stockout_costs = []

    for q in quantities:
        z = (q - forecast_p50) / max(forecast_std, 1e-6)
        exp_over   = max(0, forecast_std * (norm.pdf(z) - z * (1 - norm.cdf(z))))
        exp_under  = max(0, forecast_std * (norm.pdf(z) + z * norm.cdf(z) - z))
        hc = exp_over  * holding_cost_day
        sc = exp_under * stockout_cost
        holding_costs.append(hc)
        stockout_costs.append(sc)
        total_costs.append(hc + sc)

    return (
        np.array(quantities),
        np.array(total_costs),
        np.array(holding_costs),
        np.array(stockout_costs),
    )


def weekly_order_plan(daily_p50: np.ndarray,
                      daily_p10:  np.ndarray,
                      daily_p90:  np.ndarray,
                      unit_cost:  float,
                      selling_price: float,
                      holding_cost_rate: float = 0.20,
                      stockout_penalty:  float  = 5.0) -> Dict:
    """
    Build a 6-week order plan from daily quantile forecasts.
    Groups by week and computes optimal order per week.
    """
    weeks = {}
    n_weeks = len(daily_p50) // 7
    for w in range(n_weeks):
        sl = slice(w * 7, (w + 1) * 7)
        week_p50 = daily_p50[sl].sum()
        week_std = np.sqrt(np.sum(((daily_p90[sl] - daily_p10[sl]) / 3.29) ** 2))  # approx
        result   = newsvendor_order(week_p50, week_std, unit_cost, selling_price,
                                     holding_cost_rate, stockout_penalty)
        weeks[f'Week {w+1}'] = {
            'forecast': round(week_p50, 0),
            'order':    result['optimal_order'],
            'safety_stock': result['safety_stock'],
        }
    return weeks
