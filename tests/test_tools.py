from spectr import tools

def test_import():
    from spectr import tools
    pass

def test_AutoDict():
    t = tools.AutoDict([])
    t['x'] = [5,6]
    assert t['x'] == [5,6]
    assert t['y'] == []
    t = tools.AutoDict({})
    assert t['y'] == {}
    t['x']['a'] = 1
    assert t['x'] == {'a':1}
    ## two of them!
    t = tools.AutoDict(tools.AutoDict({}))
    assert t['x']['a'] == {}
    t['y']['b'][1] = 5
    assert t['y']['b'] == {1:5}


