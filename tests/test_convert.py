from pytest import approx
from pprint import pprint 
import numpy as np

from spectr import convert
from spectr.convert import units
from spectr.convert import difference

def test_units():
    assert units(1.,'m','m') == 1
    assert units(1.,'m','nm') == approx(1e9)
    assert units(1.,'cm','nm') == 1e7
    assert units(1.,'cm','nm','length') == 1e7
    assert units(units(1,'cm','nm'),'nm','cm') == 1
    assert units(units(1.,'cm','nm'),'cm-1','nm-1') == 1
    assert units(1.,'eV','nm','photon') == approx(1239.8419843320025)
    assert units(2,'cm-1','Hz','photon') == approx(59958491600.0)
    assert units(2,'Hz','cm-1','photon') == approx(6.67128190396304e-11)
    assert units(2.,'GHz','cm-1','photon') == approx(6.67128190396304e-2)
    assert units(2.,'GHz','cm-1') == approx(6.67128190396304e-2)

def test_difference():
    assert difference(0.1,1.,'m','m') == approx(0.1)
    assert difference(0.1,1.,'m','μm') == approx(0.1e6)
    assert difference(1e-2,5,'eV','nm','photon') == approx(0.4959367937328011,1e-5)
