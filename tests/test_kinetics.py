import pytest
from pytest import raises,approx
import numpy as np

from spectr.kinetics import *

def test_init_reaction():
    t = Reaction('N2 + photon → N + N')

def test_decode_reaction():
    t = Reaction('N2 + photon → N + N')
    assert t.reactants == ['N2','photon']
    assert t.products == ['N','N']

def test_get_rate_cofficient():
    t = Reaction('N2 + photon → N + N',formula='constant',k=1e10)
    assert t.get_rate_coefficient() == approx(1e10)
    t = Reaction('N2 + O2 → NO + NO',formula='arrhenius',A=1,B=0.5,C=0)
    assert t.get_rate_coefficient(T=300) == approx(1.)
    assert t.get_rate_coefficient(T=500) == approx(1.290994448)

test_get_rate_cofficient()
