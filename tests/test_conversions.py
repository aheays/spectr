from pprint import pprint 
import numpy as np
from spectr import conversions

# def test_homonuclear_diatomic_level_degeneracy():
    # # assert conversions.homonuclear_diatomic_level_degeneracy(J=0,Inuclear=0,sa=0) == 0
    # # assert conversions.homonuclear_diatomic_level_degeneracy(J=1,Inuclear=0,sa=1) == 3

    # ## 14N2 ground state
    # assert conversions.homonuclear_diatomic_level_degeneracy(J=0,Inuclear=1,sa=0) == 3
    # assert conversions.homonuclear_diatomic_level_degeneracy(J=1,Inuclear=1,sa=1) == 9
    # ## 15N2 ground state
    # assert conversions.homonuclear_diatomic_level_degeneracy(J=0,Inuclear=0.5,sa=0) == 3
    # assert conversions.homonuclear_diatomic_level_degeneracy(J=1,Inuclear=0.5,sa=1) == 9

    # # assert conversions.homonuclear_diatomic_level_degeneracy(J=0,Inuclear=0,sa=0) == 0
    # # assert conversions.homonuclear_diatomic_level_degeneracy(J=1,Inuclear=0,sa=0) == 0
    # # assert conversions.homonuclear_diatomic_level_degeneracy(1,0,1) == 3
    # # assert conversions.homonuclear_diatomic_level_degeneracy(0,1,1) == 6
    # # assert conversions.homonuclear_diatomic_level_degeneracy(1,1,1) == 18
    # # assert conversions.homonuclear_diatomic_level_degeneracy(0,0.5,1) == 1
    # # assert conversions.homonuclear_diatomic_level_degeneracy(1,0.5,1) == 3
    # # assert conversions.homonuclear_diatomic_level_degeneracy(1,1,1) == 18
    # # assert conversions.homonuclear_diatomic_level_degeneracy(0,1,1) == 2

