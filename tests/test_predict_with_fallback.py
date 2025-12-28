import pandas as pd
import numpy as np
from scripts.evaluate_forecasts import predict_with_fallback


class DummyModel:
    def predict(self, X):
        # return zeros of appropriate length
        return np.zeros(len(X))


def test_predict_with_fallback_missing_features(tmp_path):
    # create a small df with two categories
    df = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=4, freq='D'),
        'product_id': ['A', 'A', 'B', 'B'],
        'store_id': [1, 1, 1, 1],
        'category': ['cat1', 'cat1', 'cat2', 'cat2'],
        'units_sold': [1, 0, 2, 3],
        'future_14d_sum': [5, 4, 2, 1],
        # only include features for cat1
        'rolling_mean_7': [0.5, 0.5, np.nan, np.nan],
    })

    models = {
        'cat1': {'model': DummyModel(), 'features': ['rolling_mean_7']},
        'cat2': {'model': DummyModel(), 'features': ['rolling_mean_7','lag_7']},
    }
    global_model = DummyModel()

    df_pred, diagnostics = predict_with_fallback(df, models, global_model, strict_validation=False)

    # all rows should have preds
    assert 'pred' in df_pred.columns
    assert set(df_pred['pred_source'].unique()).issubset({'percat','global'})

    # diagnostics should report missing features for cat2
    assert any(d['category'] == 'cat2' for d in diagnostics['missing_features'])
    assert diagnostics['usage_counts']['percat'] == 2  # cat1 rows
    assert diagnostics['usage_counts']['global'] == 2  # fallback for cat2


if __name__ == '__main__':
    test_predict_with_fallback_missing_features(None)
    print('OK')