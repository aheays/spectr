from pprint import pprint 
import numpy as np
from spectr.levels import *

def test_construct():
    t = Levels()
    assert t.name == 'Levels'
    assert t['class'] == 'Levels'
    assert len(t._prototypes)>0 

def test_assignment():
    t = Levels(name='ddd')
    assert t.name == 'ddd'
    t['description'] = 'fff'
    t['notes'] = ['a','b']
    assert list(t['notes']) == ['a','b']

def test_str():
    t = Levels(name='ddd',description='fff',notes=['a','b'])
    print(t)

def test_load():
    t = Levels()
    t.load('data/levels_14N2')
    assert t['species'] == '14N2'
    assert list(t['J']) == [0,1,2,3,4]

def test_load_uncertainties():
    t = Levels()
    t.load('data/levels_14N2_with_uncertainties')
    print( t)
    assert t['species'] == '14N2'
    assert list(t['J']) == [0,1,2,3,4]
    assert np.max(np.abs(np.array(t['E'])-np.array([0.0000000,3.9791592,11.9373395,23.8742646,39.7895202])))<1e-3
    print( t['Eunc'])
    assert np.max(np.abs(np.array(t['Eunc'])-np.array(([0.1,0.1,0.2,0.2,0.1]))))<1e-3

def test_inheritance():
    t = Cinfv()
    t.load('data/levels_14N2')
    assert t['species'] == '14N2'
    assert list(t['J']) == [0,1,2,3,4]
    assert list(t['g']) == [1,3,5,7,9]

def test_load_complex_Cinfv():
    t = Cinfv()
    t.load('data/SO_rotational_levels')
    assert t.unique('species') == ['32S16O']
    assert abs(np.sum(t['E'])-3383596.8)<1


def test_plot():
    t = Cinfv()
    t.load('data/SO_rotational_levels')
    t.plot('J','E',show=False)
    
