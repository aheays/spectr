from spectr.lines import Lines

def test_construct():
    t = Lines()
    assert t.name == 'Lines'
    assert t['class'] == 'Lines'
    assert 'classpp' in t._prototypes

def test_assignment():
    t = Lines(name='ddd')
    assert t.name == 'ddd'
    t['description'] = 'fff'
    t['notes'] = ['a','b']
    assert list(t['notes']) == ['a','b']
    t['ν'] = 100.
    t['Epp'] = 150.

def test_infer_with_level_keys():
    t = Lines(Epp=100,Ep=150)
    assert t['ν'] == 50.
    t = Lines(ν=100,Ep=150)
    assert t['Epp'] == 50.
    
