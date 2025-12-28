import pandas as pd
import numpy as np
from apps.app import predict_14d_for_inventory, compute_reorder


def test_reorder_includes_supplier_fields():
    inv = pd.DataFrame({
        'product_id': ['X1','X2'],
        'store_id': ['S1','S1'],
        'category': ['Cat','Cat'],
        'units_sold': [10, 5],
        'lead_time_days': [7, 7],
        'current_on_hand': [0, 1],
        'unit_cost': [2.5, 1.0],
        'supplier_name': ['Acme Supplies', 'Beta Goods'],
        'supplier_email': ['acme@example.com', 'beta@example.com']
    })

    preds = predict_14d_for_inventory(inv, {}, None)
    reorder_df = compute_reorder(preds, hist_df=None, service_level=0.9, safety_mult=1.0)

    assert 'supplier_name' in reorder_df.columns
    assert 'supplier_email' in reorder_df.columns

    # Values should match input per SKU
    m = reorder_df.set_index('product_id')
    assert m.loc['X1','supplier_name'] == 'Acme Supplies'
    assert m.loc['X1','supplier_email'] == 'acme@example.com'
    assert m.loc['X2','supplier_name'] == 'Beta Goods'
    assert m.loc['X2','supplier_email'] == 'beta@example.com'


if __name__ == '__main__':
    test_reorder_includes_supplier_fields()
    print('OK')