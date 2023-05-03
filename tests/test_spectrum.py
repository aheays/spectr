from pprint import pprint
import pytest
from pytest import raises,approx
import numpy as np

from spectr import spectrum
from spectr import tools
from spectr import lines
from spectr import levels
from spectr import plotting
from spectr.optimise import P

make_plot =  True 

def test_init():
    experiment = spectrum.Experiment()
    model = spectrum.Model(x=np.arange(100,200,0.01))
    experiment = spectrum.Experiment(x=np.arange(100,200,0.01),
                                     y=np.arange(100,200,0.01),)
    model = spectrum.Model(experiment=experiment)
    
def test_construct_experiment():
    t = spectrum.Experiment()
    t.construct()
    assert t.x is None
    assert t.y is None

def test_construct_model():
    m = spectrum.Model(x=np.arange(1,100,0.1))
    m.construct()
    assert np.all(m.y==0)
    assert len(m.x) == len(m.y)
    # x = np.arange(1,100,0.05)
    # y = m.get_spectrum(x)
    # assert len(x) == len(y)

def test_load_exp_spectrum():
    t = spectrum.Experiment()
    x,y = tools.file_to_array_unpack('data/CS2_experimental_spectrum.h5')
    t.set_spectrum(x,y)
    t.construct()
    assert t.x[0] == approx(1532.5)
    assert len(t) == 25000
    assert t.y[0] == approx(1.0121078158037908)

def test_model_intensity():
    t = spectrum.Model(x=np.arange(5))
    t.add_constant(1)
    # t.get_spectrum(range(5))
    assert len(t.y) == 5
    assert t.x[0] == 0
    assert t.y[0] == 1
    t = spectrum.Model(x=np.arange(5))
    t.add_constant(P(1,False,1e-3))
    # t.get_spectrum(range(5))
    assert len(t.y) == 5
    assert t.x[0] == 0
    assert t.y[0] == 1

def test_residual_intensity():
    e = spectrum.Experiment(filename='data/CS2_experimental_spectrum.h5')
    t = spectrum.Model(experiment=e)
    t.add_constant(1)
    t.construct()
    assert len(t.y) == 25000
    assert np.min(t.get_residual()) == approx(-0.7982384401866275)
    assert np.max(t.get_residual()) == approx(0.041907587784343336)

def test_fit_intensity():
    e = spectrum.Experiment(filename='data/CS2_experimental_spectrum.h5')
    t = spectrum.Model(experiment=e)
    t.add_constant(P(1.1, True,1e-3))
    t.optimise()
    if make_plot:
        fig = plotting.qfig()
        t.plot()
    assert t.rms == approx(0.04259560696454527,1e-3)

def test_model_some_lines():
    linelist = lines.Generic('linelist')
    linelist.load_from_string('''
    species | Teq | ν   | τ   | Γ
    H₂O     | 300 | 100 | 0.1 | 1
    H₂O     | 300 | 110 | 0.5 | 1
    H₂O     | 300 | 115 | 2   | 3
    ''')
    mod = spectrum.Model('mod',x=np.arange(90,130,1e-2))
    mod.add_constant(1)
    mod.add_line(linelist)
    if make_plot:
        plotting.qfig()
        mod.plot()

# def test_add_absorption_lines():
    # t = spectrum.Model()
    # l = lines.TriatomicDinfh()
    # t.add_intensity(1)
    # l.load('data/CS2_linelist.rs')
    # l['Γ'],l['Nself'] = 0.1,1e16
    # t.add_absorption_lines(l)
    # t.get_spectrum(np.arange(1500,1550,0.1))
    # assert len(t) == 500
    # assert t.y.min() == approx(0.9618659534983689)

# def test_optimise_band():
    # e = spectrum.Experiment(filename='data/CS2_experimental_spectrum.h5')
    # t = spectrum.Model(experiment=e)
    # t.add_intensity(intensity=1)
    # l = lines.TriatomicDinfh()
    # l.load('data/CS2_linelist.rs')
    # l['species'] = '[12C][32S]2'
    # l.optimise_value(
        # Nself=(+1.04892151e+16, True,1e15),
        # Teq=298,
        # Γ=(1e-5, None,1e-6 ),
    # )
    # t.add_absorption_lines(l)
    # t.get_residual()
    # t.optimise(verbose=True)
    # if make_plot:
        # fig,ax = plotting.fig()
        # t.plot()
        # plotting.show()
    # assert len(t) == 25000
    # assert t.rms == approx(0.010049628082012397)

# def test_output_to_directory():
    # l = lines.TriatomicDinfh()
    # l.load('data/CS2_linelist.rs')
    # l['species'] = '[12C][32S]2'
    # l.optimise_value(
        # Nself=(+1.04892151e+16, True,1e15),
        # Teq=298,
        # Γ=(1e-5, None,1e-6 ),)
    # e = spectrum.Experiment(filename='data/CS2_experimental_spectrum.h5')
    # t = spectrum.Model(experiment=e)
    # t.add_intensity(1)
    # t.add_absorption_lines(l)
    # t.get_residual()
    # t.construct()
    # t.save_to_directory('tmp/test_spectrum',trash_existing= True)

if make_plot:
    plotting.show()
