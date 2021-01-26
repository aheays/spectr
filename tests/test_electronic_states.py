import pytest
from pytest import raises,approx
from scipy import constants

make_plot =False 

from spectr import *

## prepare test potentials
dR = 0.0005                     # intenuclear distance (Å)
R = np.arange(0,10,dR)          # internuclear distance Å
μ = 7.0015372                   # reduced mass 14N2
## harmonic
ω = 1e14
VSI = 1/2*convert(μ,'amu','kg')*ω**2*convert(R-2,'Å','m')**2
Vharmonic = convert(VSI,'J','cm-1')
ΔE = constants.hbar*ω
ωe_harmonic = convert(ΔE,'J','cm-1')
## morse
De = 50000                      # cm-1
Re = 2                          # Å
a = 2                           # Å-1
Vmorse = De*(np.exp(-2*a*(R-Re))-2*np.exp(-a*(R-Re)))+De # Morse potential (cm-1)
ν0 = convert(a,'m','Å')/(2*constants.pi)*np.sqrt(2*convert(De,'cm-1','J')/convert(μ,'amu','kg'))
ωe_morse =  convert(constants.h*ν0,'J','cm-1')
ωexe_morse =  ωe_morse**2/(4*De)


def test_find_single_channel_bound_levels_harmonic_oscillator():
    v,E,χ = electronic_states.find_single_channel_bound_levels_in_energy_range(Vharmonic,dR,μ,Emax=50000,δE=1e-5)
    Etheory = ωe_harmonic*(v+1/2)
    for Ei,Etheoryi in zip(E,Etheory):
        assert Ei == approx(Etheoryi)
    if make_plot:
        qfig()
        plot(R,Vharmonic)
        for vi,Ei,χi in zip(v,E,χ):
            plot(R,χi/χi.max()*500+Ei,color='black')
        ylim(0,50000)
        xlim(0,4)
        title('Harmonic potential')
        subplot()
        plot(v,E-Etheory)
        xlabel('v')
        ylabel('E-Etheory')


def test_find_single_channel_bound_levels_morse():
    v,E,χ = electronic_states.find_single_channel_bound_levels_in_energy_range(Vmorse,dR,μ,δE=1e-4)
    Etheory = ωe_morse*(v+1/2)-ωexe_morse*(v+1/2)**2
    for Ei,Etheoryi in zip(E,Etheory):
        assert Ei == approx(Etheoryi)
    if make_plot:
        qfig()
        plot(R,Vmorse)
        for vi,Ei,χi in zip(v,E,χ):
            plot(R,χi/χi.max()*100+Ei,color='black')
        title('Morse potential')
        ylim(0,50000)
        xlim(0,5)
        subplot()
        plot(v,E-Etheory)
        xlabel('v')
        ylabel('E-Etheory')

def test_show():
    if make_plot:
        show()
