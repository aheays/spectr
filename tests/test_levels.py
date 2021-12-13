from pprint import pprint 
import numpy as np
from spectr import levels

show_plots = False

def test_construct():
    t = levels.Base()
    assert t.name == 'base'
    assert len(t._prototypes)>0
    assert list(t._prototypes.keys()) == ['description','notes','author','reference','date','classname',]

def test_construct():
    t = levels.Generic()
    assert len(t.default_prototypes) > 0

def test_assignment():
    t = levels.Generic(name='ddd')
    assert t.name == 'ddd'
    t.description = 'fff'
    t['E'] = [1,2]
    assert all(t['E'] == [1,2])

def test_load():
    t = levels.Diatom()
    t.load('data/levels_14N2')
    assert all(t['species'] == ['14N2','14N2','14N2','14N2','14N2'])
    assert list(t['J']) == [0,1,2,3,4]
    t = levels.Diatom()
    t.load('data/SO_rotational_levels')
    assert t.unique('species') == ['[32S][16O]']
    assert abs(np.sum(t['E'])-3383596.8)<1

def test_plot():
    t = levels.Diatom()
    t.load('data/SO_rotational_levels')
    t.plot('J','E',show=show_plots)

# def test_load_HomonuclearDiatomRotational():
    # t = levels.HomonuclearDiatomRotational(
        # species='14N2',label='X',v=0,S=0,Λ=0,s=0,gu=1,Inuclear=1,J=[0,1,2,3])
    # assert set(t._prototypes) == set([ 'species', 'Eref', 'label', 'Λ',
                                  # 's', 'LSsign', 'v', 'Γv', 'τv', 'Atv', 'Adv', 'Aev', 'ηdv', 'ηev',
                                  # 'Tv', 'Bv', 'Dv', 'Hv', 'Lv', 'Av', 'ADv', 'AHv', 'λv', 'λDv',
                                  # 'λHv', 'γv', 'γDv', 'γHv', 'ov', 'oDv', 'oHv', 'oLv', 'pv', 'qv',
                                  # 'pDv', 'qDv', 'Tvreduced', 'Tvreduced_common', 'Bv_μscaled', 'E',
                                  # 'J', 'g', 'pm', 'Γ', 'N', 'S', 'Teq', 'Tex', 'partition_source',
                                  # 'partition', 'α', 'Nself', 'σv', 'sa', 'ef', 'Fi', 'Ω', 'Σ', 'SR',
                                     # 'Inuclear', 'gu','ΓD',])

def test_level_degeneracy():
    t = levels.Diatom()
    t.verbose = True
    t.load('data/SO_rotational_levels')
    assert t['g'][0] == 1
    assert t['g'][1] == 3
    t = levels.Diatom(species='[14N][15N]',label='X',v=0,S=0,Λ=0,s=0,J=[0,1,2,3])
    assert all(t['g'] == [1,3,5,7])
    t = levels.Diatom(species='[14N]2',label='X',v=0,S=0,Λ=0,s=0,gu=1,Inuclear=1,J=[0,1,2,3])
    from spectr import kinetics
    print( kinetics.get_species('[14N]2').isotopes)
    print( kinetics.get_species('[14N]2').elements)
    print( kinetics.get_species('[14N]2').point_group)
    print( t['point_group'])
    assert list(t['g']) == [6,9,30,21]
    t = levels.Diatom(species='[15N]2',label='X',v=0,S=0,Λ=0,s=0,gu=1,Inuclear=0.5, J=[0,1,2,3],)
    assert list(t['g']) == [1,9,5,21]
