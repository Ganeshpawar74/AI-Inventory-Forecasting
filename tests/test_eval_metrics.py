import numpy as np
from scripts.eval_metrics import compute_metrics


def test_compute_metrics_basic():
    y_true = np.array([10, 0, 5, 0])
    y_pred = np.array([8, 0, 7, 0])
    m = compute_metrics(y_true, y_pred)
    assert m['n'] == 4
    assert abs(m['MAE'] - (2+0+2+0)/4) < 1e-6
    assert abs(m['WAPE'] - ((2+0+2+0)/(10+0+5+0))) < 1e-6
    # Service level: sum(min(pred,true))/sum(true) = (8+0+5+0)/(10+0+5+0) = 13/15
    assert abs(m['ServiceLevel_pct'] - (13/15*100)) < 1e-6
    # Stockout rate: non-zero periods = 2, periods where pred<true -> 1 (10>8) so 0.5
    assert abs(m['StockoutRate'] - 50.0) < 1e-6


if __name__ == '__main__':
    test_compute_metrics_basic()
    print('OK')
