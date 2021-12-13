## automatically fit one spectrum with multiple species

from spectr.env import *



o = spectrum.FitAbsorption(
    filename='../scans/miniPALS/2021_11_25_HCN_723Torr_mix+N2.0',
    verbose= True,
    # parameters=parameters,
)
o.load_parameters('t0.py')

o.fit_species(
    species_to_fit=[
        # 'CO',
        # 'CO2',
        # 'HCN',
        'H2O',
    ],
    regions='bands',
    # regions='lines',
    fit_N= True,
    fit_pair= True,
    fit_intensity= True,
    fit_FTS_H2O= True,
)


# o.fit(
    # species_to_fit=['CO', 'CO2', 'HCN','H2O'],
    # # species_to_fit=['H2O',],
    # # regions='bands',
    # # regions='lines',
    # fit_N= True,
    # fit_pair= True,
    # fit_intensity= True,
    # fit_FTS_H2O= True,
# )

o.plot(fig=1)

# o.save_parameters('t0.py')
o.save_parameters('t1.py')

show()
