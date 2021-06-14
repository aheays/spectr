from spectr import *


########
## H2 ##
########
d = levels.Diatomic(species='¹H₂',Eref=0)
d.load('~/data/species/H2/lines_levels/ground_state/H2_term_values_relative_to_fundamental_komasa2011',
       labels_commented=True,
       keys=('species','label','v','J','E'),
       translate_keys={'T':'E','dT':'E_unc',},)
d.save('¹H₂.h5')

########
## Ar ##
########
t = levels.Atomic(species='Ar')
t.load_from_nist('~/data/species/Ar/lines_levels/NIST_levels_2021-06-03.tsv')
t.save('Ar.h5')

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

# ########
# ## CO ##
# ########
# # d = dataset.load('~/data/species/CO/ground_state/levels_coxon2004/all_levels.h5',classname='dataset.Dataset')
# # d.pop('Fi')
# # d['E0'] = d.pop('ZPE')
# # print('DEBUG:', len(d))
# # pprint(d.keys())
# # x = levels.Diatomic()
# # for species in d.unique('species'):
# #     print( species)
# d = levels.Diatomic()
# d.load('~/data/species/CO/ground_state/levels_coxon2004/all_levels.h5',translate_keys={'ZPE':'E0',},)
# for species in d.unique('species'):
    # d.matches(species=species).save(f'{species}.h5')

