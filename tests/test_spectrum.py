from pprint import pprint
import pytest
from pytest import raises,approx
import numpy as np

from spectr import spectrum
from spectr import tools
from spectr import lines
from spectr import levels
from spectr import plotting

make_plot = False 

def test_init():
    spectrum.Spectrum()

def test_construct_experiment():
    t = spectrum.Spectrum()
    x,y = t.construct_experiment()
    assert x is None

def test_construct_model():
    t = spectrum.Spectrum()
    x,y = t.construct_model(np.arange(1,100,0.1))
    assert len(x) == 990

def test_load_exp_spectrum():
    t = spectrum.Spectrum()
    x,y = tools.file_to_array_unpack('data/CS2_experimental_spectrum.h5')
    t.set_experimental_spectrum(x,y)
    t.construct_experiment()
    assert t.xexp[0] == approx(1532.5)
    assert len(t) == 25000
    assert t.yexp[0] == approx(1.0121078158037908)

def test_model_intensity():
    t = spectrum.Spectrum()
    t.add_intensity(1)
    t.construct_model(range(5))
    assert len(t) == 5
    assert t.xmod[0] == 0
    assert t.ymod[0] == 1
    t = spectrum.Spectrum()
    t.add_intensity((1,False,1e-3))
    t.construct_model(range(5))
    assert len(t) == 5
    assert t.xmod[0] == 0
    assert t.ymod[0] == 1

def test_residual_intensity():
    t = spectrum.Spectrum()
    x,y = tools.file_to_array_unpack('data/CS2_experimental_spectrum.h5')
    t.set_experimental_spectrum(x,y)
    t.add_intensity(1)
    t.construct_residual()
    assert len(t) == 25000
    assert t.model_residual.min() == approx(-0.7982384401866275)
    assert t.model_residual.max() == approx(0.041907587784343336)

def test_fit_intensity():
    t = spectrum.Spectrum()
    x,y = tools.file_to_array_unpack('data/CS2_experimental_spectrum.h5')
    t.set_experimental_spectrum(x,y)
    t.add_intensity((1.1, True,1e-3))
    t.optimise()
    if make_plot:
        fig,ax = plotting.fig()
        t.plot()
        plotting.show()
    assert t.rms == approx(0.04259560696454527)

def test_add_absorption_lines():
    t = spectrum.Spectrum()
    l = lines.TriatomicDinfh()
    t.add_intensity(1)
    l.load('data/CS2_linelist.rs')
    l['Γ'],l['Nself'] = 0.1,1e16
    t.add_absorption_lines(l)
    t.construct_model(np.arange(1500,1550,0.1))
    assert len(t) == 500
    assert t.ymod.min() == approx(0.9618659534983689)

def test_optimise_band():
    t = spectrum.Spectrum()
    x,y = tools.file_to_array_unpack('data/CS2_experimental_spectrum.h5')
    t.set_experimental_spectrum(x,y)
    t.add_intensity(1)
    l = lines.TriatomicDinfh()
    l.load('data/CS2_linelist.rs')
    l['species'] = '[12C][32S]2'
    l.optimise_value(
        Nself=(+1.04892151e+16, True,1e15),
        Teq=298,
        Γ=(1e-5, None,1e-6 ),
    )
    t.add_absorption_lines(l)
    t.optimise(verbose=True)
    if make_plot:
        fig,ax = plotting.fig()
        t.plot()
        plotting.show()
    assert len(t) == 25000
    assert t.rms == approx(0.010049628082012397)

