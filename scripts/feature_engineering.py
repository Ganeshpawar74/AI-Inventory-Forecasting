"""Feature engineering helpers for forecasting.

Adds rolling demand stats, zero-sales ratio, volatility, and lead-time interactions.
Functions are written to be safe when features are missing and robust to small series.
"""
from typing import List
import pandas as pd
import numpy as np


def add_rolling_features(df: pd.DataFrame, date_col: str = 'date', sku_col: str = 'product_id', store_col: str = 'store_id', demand_col: str = 'units_sold', windows: List[int] = [7,14,30]) -> pd.DataFrame:
    """Add rolling statistics named both in a generic style and in the model-expected style.

    Adds the following for each window w:
      - model-friendly names: rolling_mean_{w}, rolling_std_{w}, rolling_med_{w}
      - legacy names: rolling_{w}d_mean, rolling_{w}d_std (kept for backward-compat)
      - zero_sales_ratio_{w}d
      - vol_{w}d (coefficient of variation) and also vol_{w}

    The model-expected names (rolling_mean_7, rolling_std_30, rolling_med_30, etc.) are used
    to satisfy the feature requirements of pre-trained per-category models.
    """
    d = df.copy()
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([sku_col, store_col, date_col])

    # Compute rolling per group
    def apply_roll(g):
        g = g.set_index(date_col).asfreq('D', fill_value=0)
        for w in windows:
            # model-friendly
            mean_m = f'rolling_mean_{w}'
            std_m = f'rolling_std_{w}'
            med_m = f'rolling_med_{w}'
            # legacy
            mean_l = f'rolling_{w}d_mean'
            std_l = f'rolling_{w}d_std'
            zr_col = f'zero_sales_ratio_{w}d'
            vol_l = f'vol_{w}d'
            vol_m = f'vol_{w}'

            g[mean_l] = g[demand_col].rolling(window=w, min_periods=1).mean()
            g[mean_m] = g[mean_l]

            g[std_l] = g[demand_col].rolling(window=w, min_periods=1).std().fillna(0)
            g[std_m] = g[std_l]

            g[med_m] = g[demand_col].rolling(window=w, min_periods=1).median()
            g[med_m] = g[med_m].fillna(0)

            g[zr_col] = g[demand_col].rolling(window=w, min_periods=1).apply(lambda x: (x==0).sum()/len(x))
            # volatility as CV
            g[vol_l] = g[std_l] / g[mean_l].replace(0, np.nan)
            g[vol_m] = g[vol_l]
            g[vol_l] = g[vol_l].fillna(0)
            g[vol_m] = g[vol_m].fillna(0)

        # Reset index back
        g = g.reset_index()
        return g

    out = d.groupby([sku_col, store_col], group_keys=False).apply(apply_roll)
    out = out.reset_index(drop=True)
    return out


def add_lead_time_interactions(df: pd.DataFrame, lead_col: str = 'lead_time_days', windows: List[int] = [7,14,30]) -> pd.DataFrame:
    """Add features interacting lead-time with rolling means: lead_x_rolling_{w}
    Supports both legacy and model-friendly rolling mean names.
    """
    d = df.copy()
    for w in windows:
        # prefer model-friendly name
        mean_cols = [f'rolling_mean_{w}', f'rolling_{w}d_mean']
        # create both new forms
        for base in mean_cols:
            new_col = f'lead_x_rolling_{w}' if base.startswith('rolling_mean_') else f'lead_x_rolling_{w}d'
            if base in d.columns and lead_col in d.columns:
                d[new_col] = d[base].fillna(0) * d[lead_col].fillna(0)
            else:
                d[new_col] = 0.0
    return d


def engineer_features(df: pd.DataFrame, date_col: str = 'date', sku_col: str = 'product_id', store_col: str = 'store_id', demand_col: str = 'units_sold', lead_col: str = 'lead_time_days') -> pd.DataFrame:
    """Composite function to add standard features used by models.

    This function now generates features with names expected by the trained per-category models
    (e.g. rolling_mean_7, rolling_std_30, lag_7, rolling_med_30, day_of_week, month_sin, etc.)
    """
    d = df.copy()
    # Add rolling features (7,14,30) and model-friendly names
    d = add_rolling_features(d, date_col=date_col, sku_col=sku_col, store_col=store_col, demand_col=demand_col, windows=[7,14,30])
    # Add lag features commonly used by the models
    lags = [1,7,14,30,60,90]
    d = d.sort_values([sku_col, store_col, date_col])
    def apply_lags(g):
        g = g.set_index(date_col).asfreq('D', fill_value=0)
        for lag in lags:
            g[f'lag_{lag}'] = g[demand_col].shift(lag).fillna(0)
        return g.reset_index()
    d = d.groupby([sku_col, store_col], group_keys=False).apply(apply_lags).reset_index(drop=True)

    # Add calendar features
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col])
        d['day_of_week'] = d[date_col].dt.dayofweek
        d['is_weekend'] = d['day_of_week'].isin([5,6]).astype(int)
        d['month'] = d[date_col].dt.month
        d['quarter'] = d[date_col].dt.quarter
        # seasonal cycle encoding
        d['month_sin'] = np.sin(2 * np.pi * (d['month'] / 12.0))
        d['month_cos'] = np.cos(2 * np.pi * (d['month'] / 12.0))

    # Add lead-time interactions
    d = add_lead_time_interactions(d, lead_col=lead_col, windows=[7,14,30])

    # Add a simple demand volatility measure (max of vol_7, vol_14, vol_30)
    vol_cols = []
    for w in [7,14,30]:
        v = f'vol_{w}'
        vol_cols.append(v)
    existing_vols = [c for c in vol_cols if c in d.columns]
    d['demand_volatility'] = d[existing_vols].max(axis=1) if existing_vols else 0.0

    # Zero sales ratio over 30d (model-friendly already generated)
    if 'zero_sales_ratio_30d' in d.columns:
        d['zero_sales_ratio'] = d['zero_sales_ratio_30d']
    else:
        d['zero_sales_ratio'] = 0.0

    # unit_price / log_unit_price helper
    if 'unit_price' in d.columns:
        d['log_unit_price'] = np.log1p(d['unit_price'].clip(lower=0))
    elif 'unit_cost' in d.columns:
        d['unit_price'] = d['unit_cost']
        d['log_unit_price'] = np.log1p(d['unit_cost'].clip(lower=0))

    return d
