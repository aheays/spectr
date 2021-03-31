import pytest
from pytest import raises,approx
import numpy as np

from spectr.kinetics import *

def test_init_reaction():
    t = Reaction('N2 + photon → N + N')

def test_decode_reaction():
    t = Reaction('N2 + photon → N + N')
    assert list(t.reactants) == ['N2','photon']
    assert list(t.products) == ['N','N']

def test_get_rate_cofficient():
    t = Reaction('N2 + photon → N + N',formula='constant',coefficients={'k':1e10})
    assert t.get_rate_coefficient() == approx(1e10)
    t = Reaction('N2 + O2 → NO + NO',formula='arrhenius',coefficients={'A':1,'B':0.5,'C':0})
    assert t.get_rate_coefficient(state={'T':300}) == approx(1.)
    assert t.get_rate_coefficient(state={'T':500}) == approx(1.290994448)

def test_species_init():
    t = Species('N2')
    t = Species('[14N]2')

def test_species_sort():
    x = Species('N2')
    y = Species('[14N]2')
    assert x == x
    assert y > x

def test_species_mass():
    assert Species('[14N]2')['mass'] == approx(28.006147)
    assert Species('[14N]2')['reduced_mass'] == approx(7.0015372)

# def test_species_element():
    # assert Species('NH3')['elements'] == sorted(['H','H','H','N'])

# def test_species_charge():
    # assert Species('NH3')['charge'] == 0
    # assert Species('NH3++')['charge'] == 2
    # assert Species('NH3-')['charge'] == -1
