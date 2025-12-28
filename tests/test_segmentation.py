import pandas as pd
from scripts.segmentation import segment_skus, assign_segment_to_df


def test_segmentation_basic():
    data = pd.DataFrame([
        {'product_id':'A','store_id':'S1','date':'2025-01-01','units_sold':2},
        {'product_id':'A','store_id':'S1','date':'2025-01-02','units_sold':3},
        {'product_id':'B','store_id':'S1','date':'2025-01-01','units_sold':0},
        {'product_id':'B','store_id':'S1','date':'2025-01-02','units_sold':0},
        {'product_id':'C','store_id':'S1','date':'2025-01-01','units_sold':1},
        {'product_id':'C','store_id':'S1','date':'2025-01-02','units_sold':0},
    ])
    seg = segment_skus(data)
    # A should be fast (avg 2.5)
    a = seg[(seg['product_id']=='A') & (seg['store_id']=='S1')]
    assert a['segment'].iloc[0] == 'fast'
    b = seg[(seg['product_id']=='B') & (seg['store_id']=='S1')]
    assert b['segment'].iloc[0] == 'slow'
    c = seg[(seg['product_id']=='C') & (seg['store_id']=='S1')]
    # avg = 0.5 => medium
    assert c['segment'].iloc[0] == 'medium'


if __name__ == '__main__':
    test_segmentation_basic()
    print('OK')