from pytest import raises,approx,fixture
from spectr import *
tools.warnings_off()

show_plot =False

def test_load_model():
    tools.warnings_off()
    model = atmosphere.AtmosphericChemistry()
    model.load_argo('data/ARGO_early_earth')

def test_calc_rates():
    tools.warnings_off()
    model = atmosphere.AtmosphericChemistry()
    model.load_argo('data/ARGO_early_earth')
    model.reaction_network.calc_rates()

def test_calculate_some_things():
    tools.warnings_off()
    model = atmosphere.AtmosphericChemistry()
    model.load_argo('data/ARGO_early_earth')
    model.reaction_network.calc_rates()
    assert model.get_surface_mixing_ratio()['N2'] == approx(0.7961)
    assert model.get_surface_mixing_ratio()['CO2'] == approx(0.1989)
    total_column_density = integrate.trapz(model['nt'],model['z'])
    print( total_column_density)
    assert total_column_density == approx(2.11281e25)
    for ispecies,species in enumerate(('NO','NO2','N2O')):
        model.summarise_species(species,doprint=True )
    
def test_plot_abundances_and_ratesx():
    tools.warnings_off()
    model = atmosphere.AtmosphericChemistry()
    model.load_argo('data/ARGO_early_earth')
    model.reaction_network.calc_rates()
    model.plot_vertical('z(km)', 'n(NO)','n(NO2)','n(N2O)',ax=plotting.qax())
    plotting.qfig()
    for ispecies,species in enumerate(('NO','NO2','N2O')):
        model.plot_production_destruction(species, ykey='z(km)', normalise=False, nsort=5, ax=plotting.subplot(),)
    if show_plot:
        plotting.show()
