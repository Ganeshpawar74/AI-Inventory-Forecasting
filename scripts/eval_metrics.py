"""Evaluation metrics for forecasting with inventory focus.

Provides:
- compute_metrics: returns MAE, RMSE, WAPE, MAE/avg, Service Level (fill rate), Stockout Rate, R2
- safe_r2: computes R2 and handles negative values (documented behavior)

Notes:
- WAPE is defined as sum(|err|) / sum(|y_true|), robust when totals are large.
- Service Level (Fill Rate) is volume-based: sum(min(pred, true)) / sum(true) -> fraction of demand met by forecast.
- Stockout Rate is fraction of non-zero-demand periods where forecast < actual (i.e., periods with under-prediction).
- We do NOT use MAPE for low-demand SKUs because dividing by small values inflates the metric.
"""

from typing import Dict, Optional
import numpy as np
from sklearn.metrics import r2_score


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 (coefficient of determination).

    R^2 can be negative when the model performs worse than predicting the mean.
    We return the raw value but callers should interpret negative R^2 as "poor model".
    If the denominator is zero (constant y_true), sklearn returns -inf; we handle by returning np.nan.
    """
    try:
        val = r2_score(y_true, y_pred)
        if np.isfinite(val):
            return float(val)
        return float('nan')
    except Exception:
        return float('nan')


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute inventory-focused forecasting metrics.

    Parameters
    ----------
    y_true: actual demand (array-like, non-negative)
    y_pred: predicted demand (array-like, may be non-negative)

    Returns
    -------
    dict with keys: n, MAE, RMSE, WAPE, MAE_over_avg, ServiceLevel_pct, StockoutRate, R2
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {
            'n': 0, 'MAE': None, 'RMSE': None, 'WAPE': None, 'MAE_over_avg': None,
            'ServiceLevel_pct': None, 'StockoutRate': None, 'R2': None
        }

    # Ensure non-negative predictions and truth
    y_true_pos = np.clip(y_true, 0, None)
    y_pred_pos = np.clip(y_pred, 0, None)

    abs_err = np.abs(y_true_pos - y_pred_pos)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_true_pos - y_pred_pos) ** 2)))

    denom = np.sum(np.abs(y_true_pos))
    wape = float(np.sum(abs_err) / denom) if denom > 0 else float('nan')

    mean_true = float(np.mean(y_true_pos))
    mae_over_avg = float(mae / mean_true) if mean_true > 0 else float('nan')

    # Service Level (fill rate): fraction of demand volume covered by predictions
    sum_min = float(np.sum(np.minimum(y_pred_pos, y_true_pos)))
    service_level = float(sum_min / denom) if denom > 0 else float('nan')

    # Stockout rate: fraction of non-zero demand periods where pred < actual
    nonzero_mask = y_true_pos > 0
    nonzero_count = int(np.sum(nonzero_mask))
    if nonzero_count > 0:
        stockout_rate = float(np.sum((y_pred_pos < y_true_pos) & nonzero_mask) / nonzero_count)
    else:
        stockout_rate = float('nan')

    r2 = safe_r2(y_true_pos, y_pred_pos)

    return {
        'n': int(n),
        'MAE': mae,
        'RMSE': rmse,
        'WAPE': wape,
        'MAE_over_avg': mae_over_avg,
        'ServiceLevel_pct': service_level * 100 if not np.isnan(service_level) else float('nan'),
        'StockoutRate': stockout_rate * 100 if not np.isnan(stockout_rate) else float('nan'),
        'R2': r2
    }
