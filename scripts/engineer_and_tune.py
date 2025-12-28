"""
Engineer additional features and perform per-category LightGBM hyperparameter tuning using Optuna.
Saves: models/future14_enhanced_percat.joblib, reports/forecast_diagnostics/enhanced_percat_metrics.csv,
 and feature importance PNGs under reports/forecast_diagnostics/enhanced/<category>/

Usage: python scripts/engineer_and_tune.py --input data/ml_dataset_sample.csv --trials 20
"""
import os
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import mean_absolute_error, r2_score

sns.set(style='whitegrid')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Local imports that may require installation
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import optuna
except Exception:
    optuna = None
try:
    import xgboost as xgb
except Exception:
    xgb = None

OUT_MODELS = Path('models')
OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path('reports/forecast_diagnostics/enhanced')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='data/ml_dataset_sample.csv')
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--cutoff', type=str, default='2024-10-02')
    p.add_argument('--min-train-rows', type=int, default=200)
    p.add_argument('--category', type=str, default=None, help='Optional: single category to tune')
    return p.parse_args()


def add_enhanced_features(df):
    df = df.sort_values(['product_id','store_id','date']).copy()

    # Additional lag features
    for lag in [60,90]:
        df[f'lag_{lag}'] = df.groupby(['product_id','store_id'])['units_sold'].shift(lag)

    # Rolling medians and skew
    df['rolling_med_30'] = df.groupby(['product_id','store_id'])['units_sold'].transform(lambda s: s.shift(1).rolling(30, min_periods=1).median())
    df['rolling_skew_30'] = df.groupby(['product_id','store_id'])['units_sold'].transform(lambda s: s.shift(1).rolling(30, min_periods=1).skew()).fillna(0)

    # Longer-term trend features
    df['rolling_mean_90'] = df.groupby(['product_id','store_id'])['units_sold'].transform(lambda s: s.shift(1).rolling(90, min_periods=1).mean())

    # price elasticity: ratio of price to price_rolling_7
    df['price_to_prev7'] = df['unit_price'] / (df['price_rolling_7'].replace(0, np.nan))
    df['price_to_prev7'] = df['price_to_prev7'].fillna(1.0)

    # interaction terms
    df['price_x_promo'] = df['unit_price'] * df['promo_flag']
    df['lag7_x_pricechg'] = df['lag_7'] * df['price_change_7_pct'].fillna(0)

    # holiday-ish indicators: festival season months (Oct-Nov-Dec) and back-to-school (Aug-Sep)
    df['is_festival_season'] = df['month'].isin([10,11,12]).astype(int)
    df['is_back_to_school'] = df['month'].isin([7,8,9]).astype(int)

    # quarter*category interaction
    df['cat_quarter'] = df['category'] + '_' + df['quarter'].astype(str)

    # log price
    df['log_unit_price'] = np.log1p(df['unit_price'])

    # winsorize extremely high recent targets (but preserve target col unchanged here)
    for col in ['lag_1','lag_7','lag_14','lag_30','lag_60','lag_90','rolling_mean_90']:
        if col in df.columns:
            upper = df[col].quantile(0.995)
            lower = df[col].quantile(0.005)
            df[col] = df[col].clip(lower, upper)

    return df


def objective_lgb(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective':'regression',
        'metric':'l2',
        'verbosity':-1,
        'boosting_type':'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.2),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': 1,
        'seed':42
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    # use callbacks-based early stopping for compatibility
    try:
        booster = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    except TypeError:
        # fallback: train without early stopping
        booster = lgb.train(params, dtrain, num_boost_round=200)
    y_pred = np.expm1(booster.predict(X_val))
    val_mae = mean_absolute_error(np.expm1(y_val), y_pred)
    return val_mae


def tune_category(df_cat, features, target_col, trials=20, cutoff_pd=None):
    # time split: cutoff_pd is pd.Timestamp
    train = df_cat[df_cat['date'] < cutoff_pd]
    val = df_cat[df_cat['date'] >= cutoff_pd]
    if len(train) < 200 or len(val) < 50:
        raise ValueError('Not enough data')
    X_train = train[features].fillna(-1)
    X_val = val[features].fillna(-1)
    y_train = np.log1p(train[target_col].values)
    y_val = np.log1p(val[target_col].values)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    func = lambda trial: objective_lgb(trial, X_train, y_train, X_val, y_val)
    try:
        study.optimize(func, n_trials=trials, show_progress_bar=True)
    except Exception as e:
        logger.warning('Optuna failed during study.optimize: %s', e)
    logger.info('Best trial params (if available): %s', getattr(study, 'best_trial', None).params if len(study.trials) else None)

    # train final model on combined train+val using best params if available, else train a robust default model
    if len(study.trials) and getattr(study, 'best_trial', None) is not None:
        best = study.best_trial.params
        params = dict(best)
        params.update({'objective':'regression','metric':'l2','verbosity':-1,'seed':42})
        dtrain_all = lgb.Dataset(pd.concat([X_train, X_val]), label=np.concatenate([y_train, y_val]))
        try:
            booster = lgb.train(params, dtrain_all, num_boost_round=500, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        except Exception:
            booster = lgb.train(params, dtrain_all, num_boost_round=200)
    else:
        # fallback default
        default_params = {'objective':'regression','metric':'l2','verbosity':-1, 'num_leaves':31, 'learning_rate':0.05, 'seed':42}
        dtrain_all = lgb.Dataset(pd.concat([X_train, X_val]), label=np.concatenate([y_train, y_val]))
        booster = lgb.train(default_params, dtrain_all, num_boost_round=200)
    return booster, study


def compute_metrics(y_true, y_pred):
    mae_v = mean_absolute_error(y_true, y_pred)
    rmse_v = np.sqrt(((y_true - y_pred) ** 2).mean())
    r2_v = r2_score(y_true, y_pred)
    return mae_v, rmse_v, r2_v


def plot_feature_importance(booster, features, outpath, top_n=30):
    try:
        fi = booster.feature_importance(importance_type='gain')
        names = features
    except Exception:
        # fallback for sklearn wrapper
        fi = booster.feature_importances_
        names = features
    df_fi = pd.DataFrame({'feature':names,'importance':fi}).sort_values('importance', ascending=False).head(top_n)
    plt.figure(figsize=(8, max(4, 0.2*len(df_fi))))
    sns.barplot(x='importance', y='feature', data=df_fi, palette='viridis')
    plt.title('Feature importance')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    args = parse_args()
    if lgb is None:
        raise RuntimeError('lightgbm not installed')
    if optuna is None:
        raise RuntimeError('optuna not installed')

    df = pd.read_csv(args.input, parse_dates=['date'])
    df = add_enhanced_features(df)

    cutoff_pd = pd.to_datetime(args.cutoff)
    categories = sorted(df['category'].unique())
    # allow running for a single category (convenient for quick tests)
    if args.category:
        if args.category not in categories:
            raise ValueError(f'Category {args.category} not found in data')
        categories = [args.category]

    features = [
        'units_sold','unit_price','log_unit_price','lead_time_days','day_of_week','is_weekend','month','quarter','month_sin','month_cos',
        'lag_1','lag_7','lag_14','lag_30','lag_60','lag_90','rolling_mean_7','rolling_mean_90','rolling_med_30','rolling_std_30','rolling_skew_30',
        'price_rolling_7','price_change_7_pct','price_to_prev7','price_x_promo','lag7_x_pricechg','promo_flag','is_festival_season','is_back_to_school',
        'lead_time_shift1','lead_time_roll_7','cat_roll7','cat_roll30'
    ]
    features = [f for f in features if f in df.columns]

    results = []
    models = {}

    for cat in categories:
        df_cat = df[df['category']==cat].copy()
        logger.info('Category %s rows=%d', cat, len(df_cat))
        try:
            # train/val split
            train = df_cat[df_cat['date'] < cutoff_pd]
            test = df_cat[df_cat['date'] >= cutoff_pd]
            if len(train) < args.min_train_rows or len(test) < 50:
                logger.warning('Skipping %s: insufficient data (train=%d,test=%d)', cat, len(train), len(test))
                continue

            # run tuning
            if optuna is None:
                raise RuntimeError('optuna not installed')
            booster, study = tune_category(df_cat, features, 'future_14d_sum', trials=args.trials, cutoff_pd=cutoff_pd)

            # store model and evaluate on test
            # test features
            X_test = test[features].fillna(-1)
            y_test = test['future_14d_sum'].values
            y_pred = np.expm1(booster.predict(X_test))
            y_pred = np.clip(y_pred, 0, None)
            mae_v, rmse_v, r2_v = compute_metrics(y_test, y_pred)

            logger.info('Cat %s: MAE=%.4f RMSE=%.4f R2=%.4f', cat, mae_v, rmse_v, r2_v)
            results.append({'category':cat,'MAE':mae_v,'RMSE':rmse_v,'R2':r2_v, 'train_rows':len(train),'test_rows':len(test)})

            # ensure category output dir exists
            cat_out = OUT_DIR / cat.replace(' ','_')
            cat_out.mkdir(parents=True, exist_ok=True)

            # save model
            models[cat] = {'model':booster,'features':features,'optuna_best':getattr(study, 'best_params', None)}
            joblib.dump(models[cat], OUT_MODELS / f'future14_enhanced_{cat}.joblib')

            # save feature importance
            try:
                plot_feature_importance(booster, features, cat_out / 'feature_importance.png')
            except Exception:
                logger.warning('Failed to save feature importance for %s', cat)

            # write incremental metrics so progress is saved even if the run is interrupted
            try:
                pd.DataFrame(results).to_csv(OUT_DIR / 'enhanced_percat_metrics.csv', index=False)
                joblib.dump(models, OUT_MODELS / 'future14_enhanced_percat_models.joblib')
            except Exception:
                logger.warning('Failed to write incremental results for %s', cat)

        except Exception as e:
            logger.exception('Failed for category %s: %s', cat, e)
            # even on failure, write current progress so partial results are available
            try:
                pd.DataFrame(results).to_csv(OUT_DIR / 'enhanced_percat_metrics.csv', index=False)
                joblib.dump(models, OUT_MODELS / 'future14_enhanced_percat_models.joblib')
            except Exception:
                logger.warning('Failed to write incremental results after error for %s', cat)

    # save aggregate (final)
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR / 'enhanced_percat_metrics.csv', index=False)
    joblib.dump(models, OUT_MODELS / 'future14_enhanced_percat_models.joblib')
    logger.info('Saved enhanced models and metrics to %s', OUT_DIR)

    # Quick step: identify categories with R2 < 0.85 for ensemble attempts
    low_perf = df_res[df_res['R2'] < 0.85]['category'].tolist()
    if low_perf and xgb is not None:
        logger.info('Attempting XGBoost ensembles for low-perf categories: %s', low_perf)
        ens_results = []
        for cat in low_perf:
            try:
                df_cat = df[df['category']==cat]
                train = df_cat[df_cat['date'] < cutoff_pd]
                test = df_cat[df_cat['date'] >= cutoff_pd]
                X_train = train[features].fillna(-1)
                y_train = np.log1p(train['future_14d_sum'].values)
                X_test = test[features].fillna(-1)
                y_test = test['future_14d_sum'].values

                # simple XGBoost model (no tuning, quick)
                xgbr = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
                xgbr.fit(X_train, y_train)
                pred_x = np.expm1(xgbr.predict(X_test))
                pred_gb = np.expm1(models[cat]['model'].predict(X_test))
                ensemble_pred = 0.5*pred_x + 0.5*pred_gb
                mae_v, rmse_v, r2_v = compute_metrics(y_test, ensemble_pred)
                logger.info('Ensemble %s: MAE=%.4f R2=%.4f', cat, mae_v, r2_v)
                ens_results.append({'category':cat,'MAE_ensemble':mae_v,'R2_ensemble':r2_v})
                joblib.dump(xgbr, OUT_MODELS / f'future14_xgb_{cat}.joblib')

                # save a quick comparison plot actual vs ensemble
                plt.figure(figsize=(10,4))
                plt.plot(test['date'], y_test, label='Actual')
                plt.plot(test['date'], ensemble_pred, label='Ensemble')
                plt.title(f'{cat} Ensemble Actual vs Pred (quick)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(OUT_DIR / cat.replace(' ','_') / 'ensemble_actual_vs_pred.png')
                plt.close()

            except Exception as e:
                logger.exception('Ensemble failed for %s: %s', cat, e)
        if ens_results:
            pd.DataFrame(ens_results).to_csv(OUT_DIR / 'ensemble_results.csv', index=False)

    logger.info('Engineering and tuning complete')


if __name__ == '__main__':
    main()
