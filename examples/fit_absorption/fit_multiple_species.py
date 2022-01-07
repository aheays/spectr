

## automatically fit one spectrum with multiple species
from spectr.env import *

o = spectrum.FitAbsorption(filename='../scans/miniPALS/2021_11_25_HCN_723Torr_mix+N2.0',)
# o.load_parameters('t0.py')

## fit species one at a time for a few lines, which is faster and uses
## less memory
for ispecies,species in enumerate([
        'CO',
        'H2O',
        'CO2',
        'HCN',
    ]):
    o.fit(
        species=species,
        region='lines',
        fit_N= True,
        fit_pair= True,
        fit_intensity= True,
        fit_FTS_H2O=( True if species=='H2O' else Fixed),
        fit_instrument=False,
        fig=100+ispecies,
    )

## fit all species at the same time, which accounts for overlapping
## spectra
o.fit(
    species=('CO', 'CO2', 'HCN',),
    region='bands',
    fit_N= True,
    fit_pair= True,
    fit_intensity= True,
    fit_FTS_H2O=( True if species=='H2O' else Fixed),
    fit_instrument=False,
    fig=200,
)

## show entire spectrum
o.fit(fit_intensity=True)
o.plot(fig=300)

# o.save_parameters('t0.py')


show()


