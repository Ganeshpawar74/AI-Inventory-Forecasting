"""SKU segmentation utilities.

Classifies SKUs into Fast/Medium/Slow movers based on historical daily demand.

Rules (configurable):
- fast: avg_daily_demand >= 2
- medium: 0.5 <= avg_daily_demand < 2
- slow: avg_daily_demand < 0.5 OR zero_sales_ratio >= 0.5

Exports:
- segment_skus(df, date_col='date', sku_col='product_id', store_col='store_id', demand_col='units_sold') -> DataFrame with columns [product_id, store_id, avg_daily, zero_sales_ratio, segment]
- assign_segment_to_df(df, segments_df) -> df with 'segment' merged in
"""
from typing import Tuple
import pandas as pd
import numpy as np


def segment_skus(df: pd.DataFrame, date_col: str = 'date', sku_col: str = 'product_id', store_col: str = 'store_id', demand_col: str = 'units_sold') -> pd.DataFrame:
    """Compute segmentation per SKU-store.

    Parameters
    ----------
    df : historical transactions/time series containing daily demand
    Returns
    -------
    DataFrame with columns [product_id, store_id, avg_daily_demand, zero_sales_ratio, segment]
    """
    d = df.copy()
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col])
    # Ensure one row per day per SKU-store by grouping
    grp = d.groupby([sku_col, store_col, date_col], dropna=False)[demand_col].sum().reset_index()

    # Compute stats per sku-store
    stats = grp.groupby([sku_col, store_col]).agg(
        days_observed=('date', 'nunique'),
        total_demand=(demand_col, 'sum'),
        zero_days=(demand_col, lambda x: (x == 0).sum())
    ).reset_index()

    stats['avg_daily_demand'] = stats['total_demand'] / stats['days_observed'].replace(0, np.nan)
    stats['zero_sales_ratio'] = stats['zero_days'] / stats['days_observed'].replace(0, np.nan)

    # Segmentation rules
    def classify(row):
        avg = row['avg_daily_demand'] if not pd.isna(row['avg_daily_demand']) else 0.0
        zr = row['zero_sales_ratio'] if not pd.isna(row['zero_sales_ratio']) else 1.0
        if avg >= 2:
            return 'fast'
        if avg >= 0.5:
            return 'medium'
        # slow if avg <0.5 or zero sales are prevalent
        if avg < 0.5 or zr >= 0.5:
            return 'slow'
        return 'medium'

    stats['segment'] = stats.apply(classify, axis=1)

    return stats[[sku_col, store_col, 'avg_daily_demand', 'zero_sales_ratio', 'segment']]


def assign_segment_to_df(df: pd.DataFrame, segments_df: pd.DataFrame, sku_col: str = 'product_id', store_col: str = 'store_id') -> pd.DataFrame:
    """Merge segment information into a working dataframe (left merge)."""
    seg = segments_df[[sku_col, store_col, 'segment']].drop_duplicates()
    out = df.merge(seg, on=[sku_col, store_col], how='left')
    out['segment'] = out['segment'].fillna('unknown')
    return out
