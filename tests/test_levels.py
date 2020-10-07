from pprint import pprint 
import numpy as np
from spectr import levels

show_plots = False

def test_construct():
    t = levels.Base()
    assert t.name == 'base'
    assert len(t.prototypes)>0
    assert list(t.prototypes.keys()) == ['description', 'notes', 'author', 'reference', 'date', 'level_type', 'species', 'E', 'J', 'g', 'pm', 'Γ', 'N', 'S', 'Eref', 'Teq', 'Tex', 'partition_source', 'partition', 'α', 'Nself']

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

def test_load_DiatomicDinfh():
    t = levels.DiatomicCinfv()
    t.load('data/levels_14N2')
    assert t['species'] == '14N2'

def test_load_DiatomicCinfv():
    t = levels.DiatomicCinfv()
    t.load('data/SO_rotational_levels')
    assert t.unique('species') == ['32S16O']
    assert abs(np.sum(t['E'])-3383596.8)<1

def test_plot():
    t = levels.DiatomicCinfv()
    t.load('data/SO_rotational_levels')
    t.plot('J','E',show=show_plots)

def test_load_DiatomicDinfh():
    t = levels.DiatomicDinfh(
        species='14N2',label='X',v=0,S=0,Λ=0,s=0,gu=1,Inuclear=1,
        J=[0,1,2,3],
    )
    assert list(t.prototypes) == ['description', 'notes', 'author', 'reference', 'date', 'level_type', 'species', 'E', 'J', 'g', 'pm', 'Γ', 'N', 'S', 'Eref', 'Teq', 'Tex', 'partition_source', 'partition', 'α', 'Nself', 'label', 'v', 'Λ', 'LSsign', 's', 'σv', 'sa', 'ef', 'Fi', 'Ω', 'Σ', 'SR', 'Inuclear', 'gu']

def test_level_degeneracy_DiatomicCinfv():
    t = levels.DiatomicCinfv()
    t.verbose = True
    t.load('data/SO_rotational_levels')
    assert t['g'][0] == 1
    assert t['g'][1] == 3

def test_level_degeneracy_DiatomicDinfh():
    t = levels.DiatomicDinfh(species='14N2',label='X',v=0,S=0,Λ=0,s=0,gu=1,Inuclear=1, J=[0,1,2,3],)
    assert list(t['g']) == [6,9,30,21]
    t = levels.DiatomicDinfh(species='15N2',label='X',v=0,S=0,Λ=0,s=0,gu=1,Inuclear=0.5, J=[0,1,2,3],)
    assert list(t['g']) == [1,9,5,21]
