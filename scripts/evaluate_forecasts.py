"""Run end-to-end forecast evaluation using saved models.

This script:
- Loads models and test dataset
- Produces row-level predictions (per-cat when features available, else global)
- Computes overall, per-category, and per-SKU metrics using scripts.eval_metrics
- Writes reports to reports/accuracy_*.csv for Streamlit consumption
"""
from scripts.simulate_reorders import load_models, DATA
from scripts.eval_metrics import compute_metrics
import pandas as pd
import numpy as np
import os


def predict_with_fallback(df, models, global_model, strict_validation: bool = False):
    """Attempt per-category predictions, fall back to global model when necessary.

    Returns: (df_with_preds, diagnostics)
    diagnostics: {
        'usage_counts': {'percat': int, 'global': int, 'none': int},
        'missing_features': [{'category': str, 'missing': [str], 'rows_affected': int}],
    }
    """
    df = df.copy()
    df['pred'] = np.nan
    df['pred_source'] = 'none'

    diagnostics = {'usage_counts': {'percat': 0, 'global': 0, 'none': 0}, 'missing_features': []}

    for cat, info in models.items():
        mask = df['category'] == cat
        if not mask.any():
            continue
        feat = info.get('features') if isinstance(info, dict) else None
        model = info.get('model') if isinstance(info, dict) else info
        X = df.loc[mask]
        if feat is not None:
            missing = [c for c in feat if c not in X.columns]
            if missing:
                diagnostics['missing_features'].append({'category': cat, 'missing': missing, 'rows_affected': int(mask.sum())})
                print(f"Warning: missing features for category {cat}: {missing}. Using global for these rows.")
                continue
            Xf = X[feat].fillna(-1)
        else:
            Xf = X.drop(columns=['future_14d_sum','date'], errors='ignore').select_dtypes(include=[np.number]).fillna(-1)
        try:
            yhat = model.predict(Xf)
        except Exception as e:
            print('Model predict failed for category', cat, 'error:', e)
            yhat = np.zeros(len(Xf))
        try:
            vals = np.expm1(yhat)
        except Exception:
            vals = yhat
        df.loc[mask, 'pred'] = vals
        df.loc[mask, 'pred_source'] = 'percat'
        diagnostics['usage_counts']['percat'] += int(mask.sum())

    # global fallback
    mask = df['pred'].isna()
    if mask.any() and global_model is not None:
        Xg = df.loc[mask].select_dtypes(include=[np.number]).fillna(-1)
        try:
            yhatg = global_model.predict(Xg)
            try:
                valsg = np.expm1(yhatg)
            except Exception:
                valsg = yhatg
            df.loc[mask, 'pred'] = valsg
            df.loc[mask, 'pred_source'] = 'global'
            diagnostics['usage_counts']['global'] += int(mask.sum())
        except Exception as e:
            print('Global predict failed:', e)
            df.loc[mask, 'pred'] = 0.0
            df.loc[mask, 'pred_source'] = 'none'
            diagnostics['usage_counts']['none'] += int(mask.sum())
    else:
        # if no global model, rows still None
        diagnostics['usage_counts']['none'] += int(df['pred'].isna().sum())

    df['pred'] = df['pred'].fillna(0)

    # strict validation: if enabled and there are missing features, raise
    if strict_validation and diagnostics['missing_features']:
        raise RuntimeError(f"Strict validation failed; missing features for categories: {diagnostics['missing_features']}")

    return df, diagnostics


def evaluate():
    models, global_model = load_models()
    df = pd.read_csv(DATA, parse_dates=['date'])
    df_test = df.dropna(subset=['future_14d_sum']).copy()

    # --- SEGMENTATION & FEATURE ENGINEERING ---
    # Segment SKUs into fast/medium/slow so we can evaluate and route models differently later
    try:
        from scripts.segmentation import segment_skus, assign_segment_to_df
        seg = segment_skus(df_test, date_col='date', demand_col='units_sold')
        df_test = assign_segment_to_df(df_test, seg)
    except Exception as e:
        print('Segmentation failed:', e)
        df_test['segment'] = 'unknown'

    try:
        from scripts.feature_engineering import engineer_features
        df_test = engineer_features(df_test, date_col='date', demand_col='units_sold', lead_col='lead_time_days')
    except Exception as e:
        print('Feature engineering failed:', e)

    import argparse

    parser = argparse.ArgumentParser(description='Evaluate forecasts with optional strict feature validation')
    parser.add_argument('--strict', action='store_true', help='Fail if per-category models cannot run due to missing features')
    args, _ = parser.parse_known_args()

    print('Predicting...')
    df_pred, diagnostics = predict_with_fallback(df_test, models, global_model, strict_validation=args.strict)

    # ensure reports folder exists
    os.makedirs('reports', exist_ok=True)

    # save diagnostics reports
    usage = pd.DataFrame([diagnostics['usage_counts']])
    usage.to_csv('reports/pred_source_counts.csv', index=False)
    if diagnostics['missing_features']:
        missing_df = pd.DataFrame(diagnostics['missing_features'])
        missing_df.to_csv('reports/missing_features_per_category.csv', index=False)
    else:
        pd.DataFrame([], columns=['category','missing','rows_affected']).to_csv('reports/missing_features_per_category.csv', index=False)

    # Overall metrics
    overall = compute_metrics(df_pred['future_14d_sum'].values, df_pred['pred'].values)

    # Per-category metrics
    cats = df_pred['category'].fillna('').unique()
    rows = []
    for c in cats:
        sub = df_pred[df_pred['category'].fillna('') == c]
        if len(sub) < 1:
            continue
        m = compute_metrics(sub['future_14d_sum'].values, sub['pred'].values)
        percat_percat_count = int((sub['pred_source'] == 'percat').sum())
        percat_global_count = int((sub['pred_source'] == 'global').sum())
        m.update({'category': c, 'percat_rows': int(len(sub)), 'percat_used': percat_percat_count, 'global_fallback_used': percat_global_count})
        rows.append(m)

    percat = pd.DataFrame(rows)

    # Per SKU metrics (aggregate 14-day sums per SKU-store pair)
    sku_stats = df_pred.groupby(['product_id','store_id']).agg(
        true_14d_sum=('future_14d_sum','sum'),
        pred_14d_sum=('pred','sum')
    ).reset_index()
    sku_metrics = sku_stats.apply(lambda r: {
        'product_id': r['product_id'], 'store_id': r['store_id'],
        **compute_metrics(np.array([r['true_14d_sum']]), np.array([r['pred_14d_sum']]))
    }, axis=1)
    sku_df = pd.DataFrame(list(sku_metrics))

    # Save reports
    pd.DataFrame([overall]).to_csv('reports/accuracy_overall_inventory.csv', index=False)
    percat.to_csv('reports/accuracy_per_category_inventory.csv', index=False)
    sku_df.to_csv('reports/accuracy_per_sku_inventory.csv', index=False)

    print('Wrote reports/accuracy_overall_inventory.csv, accuracy_per_category_inventory.csv, accuracy_per_sku_inventory.csv')


if __name__ == '__main__':
    evaluate()
