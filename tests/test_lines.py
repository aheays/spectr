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

def test_load_lines():
    t = Lines()
    t.load('data/test_lines')
    assert abs(t['ν'][0]-38358.664)<1e-2
    assert len(t)==32

def test_calculate_plot_spectrum():
    t = Lines()
    t.load('data/test_lines')
    x,y = t.calculate_spectrum(xkey='ν',ykey='f')
    assert len(x)==10000
    assert abs(sum(y)-0.00903753325337632)<1e-6
    t.plot_spectrum(xkey='ν',ykey='f',show= True)
