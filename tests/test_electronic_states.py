import pytest
from pytest import raises,approx

from spectr import *


dR = 0.001                      # Å
R = np.arange(0,10,dR)          # internuclear distance Å
Vmorse = 50000*(1-np.exp(-np.sqrt(5e4/(2*50000))*(R-1)))**2 # Morse potential (cm-1)
Vharmonic = 1/2*5e5*(R-1)**2

μ = 7.0015372                   # redcued mass 14N2

show_plot = True 

def test_find_single_channel_bound_levels_harmonic_oscillator():
    v,E,χ = electronic_states.find_single_channel_bound_levels_in_energy_range(Vharmonic,dR,μ,Emax=50000)
    if show_plot:
        qfig()
        plot(R,Vharmonic)
        for vi,Ei,χi in zip(v,E,χ):
            plot(R,χi/χi.max()*500+Ei,color='black')
        ylim(0,50000)
        xlim(0,2)
        show()
test_find_single_channel_bound_levels_harmonic_oscillator()

def test_find_single_channel_bound_levels_morse():
    v,E,χ = electronic_states.find_single_channel_bound_levels_in_energy_range(Vmorse,dR,μ,)
    if show_plot:
        qfig()
        plot(R,Vmorse)
        for vi,Ei,χi in zip(v,E,χ):
            plot(R,χi/χi.max()*100+Ei,color='black')
        show()



