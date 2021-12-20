## fit one spectrum

## import spectr environment
from spectr.env import *

## load data into Experiment object, restricted to strongest HCN
## band, and plot
experiment = spectrum.Experiment(
    name='experiment',
    filename='../scans/miniPALS/2021_11_25_HCN_723Torr_mix+N2.0',
    xbeg=3200,
    xend=3400,
)
experiment.plot(fig=1)

## begin a Model object
model = spectrum.Model(
    name='model',
    experiment=experiment,
)

## auto fit background
model.auto_add_spline(xi=10,vary= True)

## add HCN lines
model.add_hitran_line(
    'HCN',
    Nchemical_species=P(1e17, True,1e12), # column density (natural isotope abundance)
    pair=P(50000, True,1),                # air-broadening pressure (Pa)
    Teq=297,                              # temperature (K)
)

## optimise "True" adjustable parameters
model.optimise()

## plot model and experiment
model.plot(fig=2)

## save new input file etc
model.save_to_directory('output_fit_manually')

## show figures
show()
