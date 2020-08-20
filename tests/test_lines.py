from pytest import approx
import numpy as np

from  spectr import lines
from spectr.levels import Levels

show_plots =False

def test_construct():
    t = lines.Lines()
    assert t.name == 'Lines'
    assert t._levels_class == Levels

test_construct()

def test_assignment():
    t = lines.Lines(name='ddd')
    assert t.name == 'ddd'
    t['description'] = 'fff'
    t['notes'] = ['a','b']
    assert list(t['notes']) == ['a','b']
    t['ν'] = 100.
    t['E_l'] = 150.

def test_infer_with_level_keys():
    t = lines.Lines(E_l=100,E_u=150)
    assert t['ν'] == 50.
    t = lines.Lines(ν=100,E_u=150)
    assert t['E_l'] == 50.
    t = lines.Lines(E_l=100,d_E_l=0.5,E_u=150,d_E_u=0.2)
    assert t['ν'] == 50.
    assert t['d_ν'] == approx(np.sqrt(0.5**2+0.2**2))

def test_load_lines():
    t = lines.Lines()
    t.load('data/test_lines')
    assert abs(t['ν'][0]-38358.664)<1e-2
    assert len(t)==32

def test_calculate_plot_spectrum():
    t = lines.Lines()
    t.load('data/test_lines')
    print( t[:5])
    print( t[:5])
    x,y = t.calculate_spectrum(xkey='ν',ykey='f')
    assert len(x)==10000
    assert sum(y) == approx(0.00903753325337632)
    t.plot_spectrum(xkey='ν',ykey='f',show=show_plots)


def test_get_key_without_level_suffix():
    assert lines._get_key_without_level_suffix('upper','x') == None
    assert lines._get_key_without_level_suffix('upper','x_u') == 'x'
    assert lines._get_key_without_level_suffix('upper','x_u') == 'x'
    assert lines._get_key_without_level_suffix('lower','x_l') == 'x'

def test_get_upper_lower_levels():
    t = lines.Lines()
    t.load('data/test_lines')
    assert t._levels_class == Levels
    u = t.get_levels('upper')
    u = t.upper_levels
    l = t.lower_levels
    assert len(u) == 32
    assert len(l) == 32
