from pprint import pprint 
import numpy as np
from spectr import levels

show_plots = False

def test_construct():
    t = levels.Base()
    assert t.name == 'base'
    assert len(t.prototypes)>0
    assert list(t.prototypes.keys()) == ['description','notes','author','reference','date','classname',]

def test_construct():
    t = levels.GenericLevel()
    assert t.name == 'generic_level'
    assert len(t.prototypes)>0
    assert list(t.prototypes.keys()) == [
        'species',
        'E','Eref',
        'Γ','ΓD',
        'g',
        'Teq','Tex','partition_source','partition','α',
        'Nself',
        # 'species', 'E', 'Eref','Γ','ΓD',
    ]


def test_assignment():
    t = levels.GenericLevel(name='ddd')
    assert t.name == 'ddd'
    t.description = 'fff'
    t['E'] = [1,2]
    assert all(t['E'] == [1,2])

def test_load():
    t = levels.HomonuclearDiatomicRotationalLevel()
    t.load('data/levels_14N2')
    assert all(t['species'] == ['14N2','14N2','14N2','14N2','14N2'])
    assert list(t['J']) == [0,1,2,3,4]

    t = levels.HeteronuclearDiatomicRotationalLevel()
    t.load('data/SO_rotational_levels')
    assert t.unique('species') == ['32S16O']
    assert abs(np.sum(t['E'])-3383596.8)<1

def test_plot():
    t = levels.HeteronuclearDiatomicRotationalLevel()
    t.load('data/SO_rotational_levels')
    t.plot('J','E',show=show_plots)

def test_load_HomonuclearDiatomicRotationalLevel():
    t = levels.HomonuclearDiatomicRotationalLevel(
        species='14N2',label='X',v=0,S=0,Λ=0,s=0,gu=1,Inuclear=1,
        J=[0,1,2,3],)
    assert set(t.prototypes) == set([ 'species', 'Eref', 'label', 'Λ',
                                  's', 'LSsign', 'v', 'Γv', 'τv', 'Atv', 'Adv', 'Aev', 'ηdv', 'ηev',
                                  'Tv', 'Bv', 'Dv', 'Hv', 'Lv', 'Av', 'ADv', 'AHv', 'λv', 'λDv',
                                  'λHv', 'γv', 'γDv', 'γHv', 'ov', 'oDv', 'oHv', 'oLv', 'pv', 'qv',
                                  'pDv', 'qDv', 'Tvreduced', 'Tvreduced_common', 'Bv_μscaled', 'E',
                                  'J', 'g', 'pm', 'Γ', 'N', 'S', 'Teq', 'Tex', 'partition_source',
                                  'partition', 'α', 'Nself', 'σv', 'sa', 'ef', 'Fi', 'Ω', 'Σ', 'SR',
                                     'Inuclear', 'gu','ΓD',])

def test_level_degeneracy():
    t = levels.HeteronuclearDiatomicRotationalLevel()
    t.verbose = True
    t.load('data/SO_rotational_levels')
    assert t['g'][0] == 1
    assert t['g'][1] == 3
    t = levels.HeteronuclearDiatomicRotationalLevel(species='14N15N',label='X',v=0,S=0,Λ=0,s=0,J=[0,1,2,3])
    assert all(t['g'] == [1,3,5,7])
    t = levels.HomonuclearDiatomicRotationalLevel(species='14N2',label='X',v=0,S=0,Λ=0,s=0,gu=1,Inuclear=1,J=[0,1,2,3])
    print('DEBUG:', t['J'])
    print('DEBUG:', t['g'])
    assert list(t['g']) == [6,9,30,21]
    # t = levels.HomonuclearDiatomicRotationalLevel(species='15N2',label='X',v=0,S=0,Λ=0,s=0,gu=1,Inuclear=0.5, J=[0,1,2,3],)
    # assert list(t['g']) == [1,9,5,21]
