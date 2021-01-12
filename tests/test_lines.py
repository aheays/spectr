from pytest import approx
import numpy as np

from spectr import lines
from spectr import levels
from spectr import plotting

show_plots =False

def test_construct():
    t = lines.Generic()
    # assert t.name == 'generic_line'

def test_assignment():
    t = lines.Generic(name='ddd')
    assert t.name == 'ddd'
    t.description = 'fff'
    t['ν'] = [1,2]
    assert all(t['ν'] == [1,2])
    t['ν'] = 100.
    t['E_l'] = 150.

def test_infer_with_level_keys():
    t = lines.Generic(E_l=[100],E_u=[150])
    assert t['E_l'] == [100.]
    assert t['E_u'] == [150.]
    assert t['ν'] == [50.]
    t = lines.Generic(ν=[100],E_u=[150])
    assert t['E_l'] == [50.]
    t = lines.Generic(E_l=[100],unc_E_l=[0.5],E_u=[150],unc_E_u=[0.2])
    t.set_uncertainty('E_l',[0.5])
    t.set_uncertainty('E_u',[0.2])
    assert t['ν'] == [50.]
    assert t.get_uncertainty('ν') == [approx(np.sqrt(0.5**2+0.2**2))]

def test_load_lines():
    t = lines.Generic()
    t.load('data/test_lines')
    assert abs(t['ν'][0]-38358.664)<1e-2
    assert len(t)==32

def test_calculate_plot_spectrum():
    t = lines.Generic()
    t.load('data/test_lines')
    x,y = t.calculate_spectrum(xkey='ν',ykey='f')
    assert len(x)==10000
    assert sum(y) == approx(0.00903753325337632)
    t.plot_spectrum(xkey='ν',ykey='f')
    if show_plots:
        plotting.show()

def test_get_upper_lower_levels():
    t = lines.Generic()
    t.load('data/test_lines')
    assert t.description == "Fitted rovibronic transitions."
    u = t.get_upper_level()
    l = t.get_lower_level()
    assert len(u) == 32
    assert len(l) == 32
