## Load, print, and plot various things from an ARGO atmospheric model
## output directory.

from spectr.env import *
tools.warnings_off()

## load model output
model = atmosphere.AtmosphericChemistry()
model.load_argo('data/ARGO_early_earth')
model.reaction_network.calc_rates()

## Compute total atmosphere column density
print(f"total column density:    {integrate(model['nt'],model['z']):10.3e}")

## list surface mixing ratios
print( )
print('surface mixing ratio:')
for i,(a,b) in enumerate(model.get_surface_mixing_ratio().items()):
    if i>10: 
        break
    print(f'{a:20} {b:10.3e}')

## plot abundances
ax = qax(1)
model.plot_vertical('z(km)', 'n(NO)','n(NO2)','n(N2O)',ax=ax)

## compare with another model
model2 = atmosphere.AtmosphericChemistry()
model2.load_argo('data/ARGO_early_earth_with_impacts')
model2.reaction_network.calc_rates()
model2.plot_vertical('z(km)', 'n(NO)','n(NO2)','n(N2O)',linestyle=':',ax=ax)

qfig()
for ispecies,species in enumerate((
        'NO',
        'NO2',
        # 'N2O',
)):
    model.plot_production_destruction(
        species, 
        ykey='z(km)',
        normalise=False,
        nsort=2,
        ax=subplot(),
    )
    xlim(1e-20,1e4)
    print('\nsummarise species:')
    model.summarise_species(species,doprint=True )

show()
