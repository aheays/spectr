from pprint import pprint 
import numpy as np
from spectr import levels

show_plots = False 

def test_construct():
    t = levels.Base()
    assert t.name == 'Base'
    assert len(t.prototypes)>0 

def test_assignment():
    t = levels.Base(name='ddd')
    assert t.name == 'ddd'
    t['description'] = 'fff'
    t['notes'] = ['a','b']
    assert list(t['notes']) == ['a','b']

def test_load():
    t = levels.Base()
    t.load('data/levels_14N2')
    assert t['species'] == '14N2'
    assert list(t['J']) == [0,1,2,3,4]

def test_inheritance():
    t = levels.HeteronuclearDiatomic()
    t.load('data/levels_14N2')
    assert t['species'] == '14N2'
    assert list(t['J']) == [0,1,2,3,4]
    assert list(t['g']) == [1,3,5,7,9]

def test_load_HeteronuclearDiatomic():
    t = levels.HeteronuclearDiatomic()
    t.load('data/SO_rotational_levels')
    assert t.unique('species') == ['32S16O']
    assert abs(np.sum(t['E'])-3383596.8)<1

def test_plot():
    t = levels.HeteronuclearDiatomic()
    t.load('data/SO_rotational_levels')
    t.plot('J','E',show=show_plots)
    
