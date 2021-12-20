## automatically fit one spectrum with multiple species

from spectr.env import *



print('DEBUG: Timing initiated') ; import time; timer=time.perf_counter()

o = spectrum.FitAbsorption(filename='../scans/miniPALS/2021_11_25_HCN_723Torr_mix+N2.0',)
# o.load_parameters('t0.py')


for ispecies,species in enumerate([
        'CO',
        'H2O',
        'CO2',
        'HCN',
    ]):
    o.fit(
        species=species,
        region='bands',
        # region='lines',
        fit_N= True,
        fit_pair= True,
        fit_intensity= True,
        fit_FTS_H2O=( True if species=='H2O' else Fixed),
    )
    o.plot(fig=100+ispecies)

o.fit(
    # region='full',
    region=(1500,2000),
    fit_N=False,
    fit_pair=False,
    fit_intensity=False,
    fit_FTS_H2O=False,
)
o.plot(fig=1)

o.save_parameters('t0.py')

print('DEBUG: Timing elapsed',format(time.perf_counter()-timer,'12.6f'))

show()

