from pprint import pprint 
import numpy as np
from spectr.levels import *

show_plots = False 

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

def test_load():
    t = Levels()
    t.load('data/levels_14N2')
    assert t['species'] == '14N2'
    assert list(t['J']) == [0,1,2,3,4]

def test_inheritance():
    t = HeteronuclearDiatomicLevels()
    t.load('data/levels_14N2')
    assert t['species'] == '14N2'
    assert list(t['J']) == [0,1,2,3,4]
    assert list(t['g']) == [1,3,5,7,9]

def test_load_complex_HeteronuclearDiatomicLevels():
    t = HeteronuclearDiatomicLevels()
    t.load('data/SO_rotational_levels')
    assert t.unique('species') == ['32S16O']
    assert abs(np.sum(t['E'])-3383596.8)<1

def test_plot():
    t = HeteronuclearDiatomicLevels()
    t.load('data/SO_rotational_levels')
    t.plot('J','E',show=show_plots)
    
