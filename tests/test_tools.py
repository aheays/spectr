from spectr import tools

def test_import():
    from spectr import tools
    pass

def test_DictOfLists():
    t = tools.DictOfLists()
    t['x'] = (5,6,)
    assert t['x'] == [5,6]
    assert t['y'] == []

# test_import()

