from spectr import *


# ########
# ## H2 ##
# ########
# d = levels.Diatomic(species='H2',Eref=0)
# d.load('~/data/species/H2/lines_levels/ground_state/H2_term_values_relative_to_fundamental_komasa2011',
#        labels_commented=True,
#        keys=('species','label','v','J','E'),
#        translate_keys={'T':'E','dT':'E_unc',},)
# # d.save('H2.h5')




########
## N2 ##
########
for species,fn in (
        ('¹⁴N₂','~/data/species/N2/lines_levels/ground_state/le_roy2006/14N2_relative_to_equilibrium'),
        ('¹⁴N¹⁵N','~/data/species/N2/lines_levels/ground_state/le_roy2006/14N15N_relative_to_equilibrium'),
        ('¹⁵N₂','~/data/species/N2/lines_levels/ground_state/le_roy2006/15N2_relative_to_equilibrium'),
        ):
    t = dataset.load(fn,labels_commented=True )
    d = levels.Diatomic(species=species, label='X',
                        Ee=t['T'], J=t['J'], v=t['v'],
                        E0=np.min(t['T']), Eref=0,)
    d.save(f'{species}.h5')

