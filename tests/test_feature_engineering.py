import pandas as pd
from scripts.feature_engineering import engineer_features


def test_engineer_basic():
    data = pd.DataFrame([
        {'product_id':'A','store_id':'S1','date':'2025-01-01','units_sold':2,'lead_time_days':3},
        {'product_id':'A','store_id':'S1','date':'2025-01-02','units_sold':3,'lead_time_days':3},
        {'product_id':'A','store_id':'S1','date':'2025-01-03','units_sold':1,'lead_time_days':3},
        {'product_id':'A','store_id':'S1','date':'2025-01-04','units_sold':0,'lead_time_days':3},
        {'product_id':'A','store_id':'S1','date':'2025-01-05','units_sold':4,'lead_time_days':3},
    ])
    out = engineer_features(data)
    assert 'rolling_7d_mean' in out.columns
    assert 'rolling_14d_mean' in out.columns
    assert 'lead_x_rolling_7d' in out.columns
    assert 'zero_sales_ratio' in out.columns

if __name__ == '__main__':
    test_engineer_basic()
    print('OK')