"""Train baseline LightGBM models per category (and pooled global) on ML-ready dataset.
Saves models and per-category metrics.

Usage:
  python scripts/train_baseline_models.py --input data/ml_dataset_sample.csv --target future_14d_sum
"""
import os
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
REPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports'))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='data/ml_dataset_sample.csv')
    p.add_argument('--target', type=str, default='future_14d_sum')
    p.add_argument('--test-horizon-days', type=int, default=90)
    p.add_argument('--min-train-rows', type=int, default=200)
    return p.parse_args()


def _get_features(df):
    candidate = [
        'units_sold','unit_price','lead_time_days','day_of_week','is_weekend','month','quarter','month_sin','month_cos',
        'lag_1','lag_7','lag_14','lag_30','rolling_mean_7','rolling_sum_30','rolling_std_30',
        'price_rolling_7','price_change_7_pct','promo_flag','lead_time_shift1','lead_time_roll_7','cat_roll7','cat_roll30'
    ]
    return [c for c in candidate if c in df.columns]


def try_import_lightgbm():
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        return LGBMRegressor
    except Exception as e:
        raise ImportError('lightgbm is required to run this script')


def train_model(X_train, y_train_log, X_val, y_val_log):
    LGBMRegressor = try_import_lightgbm()
    # smaller and faster defaults with regularization
    model = LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=7, colsample_bytree=0.8, subsample=0.8)
    # try sklearn fit with early stopping first
    try:
        model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], early_stopping_rounds=50, verbose=False)
        return model
    except TypeError:
        # try sklearn fit with callbacks if early_stopping_rounds unsupported
        try:
            import lightgbm as lgb
            model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], callbacks=[lgb.early_stopping(50)])
            return model
        except Exception:
            # fallback to low-level lgb.train using callbacks
            import lightgbm as lgb
            train_set = lgb.Dataset(X_train, label=y_train_log)
            val_set = lgb.Dataset(X_val, label=y_val_log, reference=train_set)
            params = dict(objective='regression', metric='l2', learning_rate=0.05, num_leaves=31, seed=42)
            # try callbacks-based early stopping
            try:
                booster = lgb.train(params, train_set, num_boost_round=500, valid_sets=[val_set], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            except Exception:
                # last-resort: train without early stopping
                booster = lgb.train(params, train_set, num_boost_round=200)

            # simple wrapper to provide predict method
            class BoosterWrapper:
                def __init__(self, booster):
                    self.booster = booster
                def predict(self, X):
                    return self.booster.predict(X)
            return BoosterWrapper(booster)
    except Exception:
        # last resort, fit without early stopping to ensure progress
        model.fit(X_train, y_train_log)
        return model


def main(input_path, target_col, test_horizon_days, min_train_rows):
    print('Loading', input_path)
    df = pd.read_csv(input_path, parse_dates=['date'])
    cutoff = df['date'].max() - pd.Timedelta(days=test_horizon_days)
    print('Using time cutoff for test:', cutoff.date())

    features = _get_features(df)
    print('Features used:', features)

    categories = sorted(df['category'].dropna().unique())
    results = []
    models = {}

    for cat in categories:
        df_cat = df[df['category']==cat].copy()
        n_rows = len(df_cat)
        print(f'Training category {cat}: {n_rows} rows')
        # split
        train = df_cat[df_cat['date'] < cutoff]
        test = df_cat[df_cat['date'] >= cutoff]
        if len(train) < min_train_rows or len(test) < 50:
            print(f'  Skipping {cat}: not enough train/test rows (train={len(train)}, test={len(test)})')
            results.append({'category':cat,'status':'skipped','train_rows':len(train),'test_rows':len(test)})
            continue

        X_train = train[features].fillna(-1)
        X_test = test[features].fillna(-1)
        y_train = train[target_col].values
        y_test = test[target_col].values

        # transform target
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)

        try:
            model = train_model(X_train, y_train_log, X_test, y_test_log)

            # predict and invert
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            y_pred = np.clip(y_pred, 0, None)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f'  {cat}: MAE={mae:.4f}, R2={r2:.4f}')

            results.append({'category':cat,'status':'trained','train_rows':len(train),'test_rows':len(test),'mae':mae,'r2':r2})
            models[cat] = {'model':model,'features':features}
        except Exception as e:
            print(f'  ERROR training {cat}:', str(e))
            results.append({'category':cat,'status':'error','error':str(e),'train_rows':len(train),'test_rows':len(test)})

    # pooled global model
    train_global = df[df['date'] < cutoff]
    test_global = df[df['date'] >= cutoff]
    print('Training pooled global model, train rows:', len(train_global), 'test rows:', len(test_global))
    Xg = train_global[features].fillna(-1)
    Xg_test = test_global[features].fillna(-1)
    yg = np.log1p(train_global[target_col].values)
    yg_test = np.log1p(test_global[target_col].values)
    model_global = train_model(Xg, yg, Xg_test, yg_test)
    pred_global = np.expm1(model_global.predict(Xg_test))
    pred_global = np.clip(pred_global, 0, None)
    mae_g = mean_absolute_error(test_global[target_col].values, pred_global)
    r2_g = r2_score(test_global[target_col].values, pred_global)
    print(f'Global model: MAE={mae_g:.4f}, R2={r2_g:.4f}')

    # Save models and metrics
    joblib.dump(models, os.path.join(MODEL_DIR, f'{target_col}_percat_models.joblib'))
    joblib.dump(model_global, os.path.join(MODEL_DIR, f'{target_col}_global_model.joblib'))

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(REPORT_DIR, f'forecast_baseline_{target_col}_percat_metrics.csv'), index=False)

    with open(os.path.join(REPORT_DIR, f'forecast_baseline_{target_col}_summary.txt'), 'w') as f:
        f.write(f'Global MAE: {mae_g:.4f}, R2: {r2_g:.4f}\n')

    print('Saved models and reports to', MODEL_DIR, REPORT_DIR)


if __name__ == '__main__':
    args = parse_args()
    main(args.input, args.target, args.test_horizon_days, args.min_train_rows)
