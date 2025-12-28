from apps.app import make_mailto


def test_make_mailto_basic():
    html = make_mailto('supplier@example.com')
    assert 'mailto:supplier@example.com' in html
    assert '>supplier@example.com<' in html


def test_make_mailto_with_name():
    html = make_mailto('supplier@example.com', 'Acme Supplies')
    assert 'mailto:supplier@example.com' in html
    assert '>Acme Supplies<' in html


if __name__ == '__main__':
    test_make_mailto_basic()
    test_make_mailto_with_name()
    print('OK')