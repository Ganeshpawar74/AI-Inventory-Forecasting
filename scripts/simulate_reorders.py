"""
Compute per-SKU safety stock (sigma_LT) from forecast residuals and run a simple
cost-aware reorder simulation to produce per-SKU reorder suggestions.

Outputs:
- reports/reorder_suggestions_<service_level>.csv with columns:
  product_id, store_id, current_on_hand, on_order, lead_time_days, ROP, safety_stock,
  suggested_order_qty, suggested_order_date, service_level, dormant, reorder_yes

Assumptions (for demo):
- Uses per-category enhanced 14-day models when available, else global model.
- Uses historical test residuals (future_14d_sum - pred) to estimate std of 14-day sums,
  converts to daily sigma and then to sigma over lead time.
- If there is no inventory data, sets current_on_hand = round(pred_daily * lead_time_days * 0.75)
  and on_order = 0 by default. These can be adjusted via CLI later.
"""

import argparse
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'ml_dataset_sample.csv'
SYN = ROOT / 'data' / 'synthetic_inventory_inr.csv'
MODELS_DIR = ROOT / 'models'
OUT_DIR = ROOT / 'reports'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_models():
    # Load per-category models dict if present
    models = {}
    percat_path = MODELS_DIR / 'future14_enhanced_percat_models.joblib'
    if percat_path.exists():
        try:
            models = joblib.load(percat_path)
        except Exception:
            # Fall back to individual files
            models = {}
    # Also look for individual per-category files
    for p in MODELS_DIR.glob('future14_enhanced_*.joblib'):
        cat = p.stem.replace('future14_enhanced_', '')
        try:
            m = joblib.load(p)
            # If file is booster, wrap
            if not isinstance(m, dict):
                models[cat] = {'model': m, 'features': None}
            else:
                models[cat] = m
        except Exception:
            continue
    # Load global model
    global_model = None
    gm = MODELS_DIR / 'future_14d_sum_global_model.joblib'
    if gm.exists():
        global_model = joblib.load(gm)
    return models, global_model


def compute_residual_stats(df_test, models, global_model, target_col='future_14d_sum'):
    # For each row in df_test, compute prediction (per-cat if available)
    df = df_test.copy()
    df['pred'] = np.nan

    for cat, info in models.items():
        mask = df['category'] == cat
        if not mask.any():
            continue
        feat = info.get('features') if isinstance(info, dict) else None
        model = info.get('model') if isinstance(info, dict) else info
        X = df.loc[mask]
        # If model expects specific features but they are missing from the dataset,
        # skip per-category prediction and allow global fallback later.
        if feat is not None:
            missing = [c for c in feat if c not in X.columns]
            if missing:
                print(f"Warning: missing features for category {cat}: {missing}. Using global model fallback for these rows.")
                continue
            Xf = X[feat].fillna(-1)
        else:
            Xf = X.drop(columns=[target_col, 'date'], errors='ignore').select_dtypes(include=[np.number]).fillna(-1)
        try:
            yhat = model.predict(Xf)
        except Exception:
            # Try LightGBM booster or other shapes
            try:
                yhat = model.predict(Xf)
            except Exception:
                yhat = np.zeros(len(Xf))
        # Some models predict log1p; we attempt to invert safely
        try:
            df.loc[mask, 'pred'] = np.expm1(yhat)
        except Exception:
            df.loc[mask, 'pred'] = yhat

    # Global fallback
    mask = df['pred'].isna()
    if mask.any() and global_model is not None:
        Xg = df.loc[mask].select_dtypes(include=[np.number]).fillna(-1)
        try:
            yhatg = global_model.predict(Xg)
            df.loc[mask, 'pred'] = np.expm1(yhatg)
        except Exception:
            df.loc[mask, 'pred'] = 0.0

    # Clip negative and compute residuals
    df['pred'] = df['pred'].clip(0)
    df['resid_14d'] = df[target_col] - df['pred']

    # Per SKU residual std (std of 14-day sums)
    sku_stats = df.groupby(['product_id', 'store_id', 'lead_time_days']).agg(
        pred_14d_mean=('pred', 'mean'),
        pred_14d_med=('pred', 'median'),
        resid_14d_std=('resid_14d', 'std'),
        count=('resid_14d', 'count')
    ).reset_index()
    return sku_stats


def calibrate_and_simulate(sku_stats, service_levels=[0.9, 0.95], current_on_hand_factor=0.75, assume_on_order=0, dormant_min_count=5, dormant_min_daily=0.01):
    out_rows = []
    for _, r in sku_stats.iterrows():
        product_id = r['product_id']
        store_id = r['store_id']
        lead = int(r['lead_time_days']) if not np.isnan(r['lead_time_days']) else 7
        pred14 = float(r['pred_14d_mean']) if not np.isnan(r['pred_14d_mean']) else 0.0
        std14 = float(r['resid_14d_std']) if not np.isnan(r['resid_14d_std']) else 0.0

        # Convert std of 14-day sums to per-day std: std_daily = std14 / sqrt(14)
        std_daily = std14 / math.sqrt(14) if std14 > 0 else 0.0
        # sigma over lead time
        sigma_LT = std_daily * math.sqrt(lead)
        pred_daily = pred14 / 14.0
        expected_LT = pred_daily * lead

        current_on_hand = int(round(pred_daily * lead * current_on_hand_factor))
        on_order = int(assume_on_order)

        # Dormant detection: too few history points or near-zero predicted daily demand or zero mean prediction
        count_val = int(r['count']) if not np.isnan(r['count']) else 0
        dormant = (count_val < dormant_min_count) or (pred_daily < dormant_min_daily) or (pred14 == 0)

        for sl in service_levels:
            z = norm.ppf(sl)
            safety_stock = z * sigma_LT
            ROP = expected_LT + safety_stock
            suggested_order_qty = max(0, int(math.ceil(ROP - (current_on_hand + on_order))))

            # If SKU is dormant, suppress ordering suggestions
            if dormant:
                suggested_order_qty = 0
                suggested_date = ''
            else:
                suggested_date = pd.Timestamp.today().strftime('%Y-%m-%d') if suggested_order_qty > 0 else ''

            reorder_yes = (suggested_order_qty > 0) and (not dormant)

            out_rows.append({
                'product_id': product_id,
                'store_id': store_id,
                'lead_time_days': lead,
                'current_on_hand': current_on_hand,
                'on_order': on_order,
                'service_level': sl,
                'z': float(z),
                'pred_14d_mean': pred14,
                'pred_daily': pred_daily,
                'expected_LT': expected_LT,
                'sigma_LT': sigma_LT,
                'safety_stock': float(safety_stock),
                'ROP': float(ROP),
                'suggested_order_qty': suggested_order_qty,
                'suggested_order_date': suggested_date,
                'dormant': bool(dormant),
                'reorder_yes': bool(reorder_yes)
            })
    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(DATA), help='ml dataset sample csv')
    parser.add_argument('--service-levels', default='0.9,0.95', help='comma separated service levels')
    parser.add_argument('--out', default=str(OUT_DIR), help='output folder for CSVs')
    parser.add_argument('--dormant-min-count', type=int, default=5, help='Minimum residual-count to consider SKU active')
    parser.add_argument('--dormant-min-daily', type=float, default=0.01, help='Minimum predicted daily demand to consider SKU active')

    args = parser.parse_args()

    svc = [float(x) for x in args.service_levels.split(',')]

    print('Loading models...')
    models, global_model = load_models()
    print('Loading dataset (this may take a while)...')
    df = pd.read_csv(args.input, parse_dates=['date'])

    # identify test rows where future_14d_sum is available (dropped end rows in prepare script)
    df_test = df.dropna(subset=['future_14d_sum']).copy()
    print('Computing residual stats...')
    sku_stats = compute_residual_stats(df_test, models, global_model)

    print('Calibrating safety stock and simulating reorders...')
    df_out = calibrate_and_simulate(
        sku_stats,
        service_levels=svc,
        current_on_hand_factor=0.75,
        assume_on_order=0,
        dormant_min_count=args.dormant_min_count,
        dormant_min_daily=args.dormant_min_daily
    )
    out_folder = Path(args.out)
    for sl in svc:
        outp = out_folder / f'reorder_suggestions_{int(sl*100)}pct.csv'
        df_out[df_out['service_level'] == sl].to_csv(outp, index=False)
        print('Wrote', outp)

    print('Done')


if __name__ == '__main__':
    main()
