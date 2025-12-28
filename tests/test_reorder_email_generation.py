from apps.app import generate_reorder_email

def test_generate_reorder_email_basic():
    row = {
        'product_id': 'P1', 'store_id': 'S1', 'suggested_order_qty': 10,
        'ROP': 15.5, 'stock_position': 2, 'Forecast_14d': 14.0, 'lead_time_days': 7, 'supplier_name': 'Acme'
    }
    subj, body = generate_reorder_email(row, from_email='procurement@example.com', company_name='Acme Retail')
    assert 'P1' in subj
    assert '10' in body
    assert 'Current stock position' in body

if __name__ == '__main__':
    test_generate_reorder_email_basic()
    print('OK')