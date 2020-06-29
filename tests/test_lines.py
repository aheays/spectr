from pytest import approx
import numpy as np

from spectr.lines import Lines
from spectr.levels import Levels

show_plots =False

def test_construct():
    t = Lines()
    assert t.name == 'Lines'
    assert t['class'] == 'Lines'
    assert t['levels_class'] == 'Levels'

def test_assignment():
    t = Lines(name='ddd')
    assert t.name == 'ddd'
    t['description'] = 'fff'
    t['notes'] = ['a','b']
    assert list(t['notes']) == ['a','b']
    t['ν'] = 100.
    t['El'] = 150.

def test_infer_with_level_keys():
    t = Lines(El=100,Eu=150)
    assert t['ν'] == 50.
    t = Lines(ν=100,Eu=150)
    assert t['El'] == 50.
    t = Lines(El=100,σEl=0.5,Eu=150,σEu=0.2)
    assert t['ν'] == 50.
    assert t['σν'] == approx(np.sqrt(0.5**2+0.2**2))

def test_load_lines():
    t = Lines()
    t.load('data/test_lines')
    assert abs(t['ν'][0]-38358.664)<1e-2
    assert len(t)==32

def test_calculate_plot_spectrum():
    t = Lines()
    t.load('data/test_lines')
    print( t[:5])
    print( t[:5])
    x,y = t.calculate_spectrum(xkey='ν',ykey='f')
    assert len(x)==10000
    assert sum(y) == approx(0.00903753325337632)
    t.plot_spectrum(xkey='ν',ykey='f',show=show_plots)

def test_get_upper_lower_levels():
    t = Lines()
    t.load('data/test_lines')
    assert t['levels_class'] == 'Levels'
    u = t.get_levels('upper')
    u = t.upper_levels
    l = t.lower_levels
