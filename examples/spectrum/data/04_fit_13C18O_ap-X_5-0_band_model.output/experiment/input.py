from spectr import *

experiment = spectrum.Experiment(name='experiment')
experiment.set_spectrum_from_soleil_file(
    filename='data/13C18O_spectrum.h5',
    xbeg=60700,
    xend=61000,
)