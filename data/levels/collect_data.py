from spectr.env import *

########
## H2 ##
########
d = levels.Diatom(species='¹H₂',Eref=0)
d.load('~/data/species/H2/lines_levels/ground_state/H2_term_values_relative_to_fundamental_komasa2011',
       labels_commented=True,
       keys=('species','label','v','J','E'),
       translate_keys={'T':'E','dT':'E_unc',},)
d['species'] = '¹H₂'
d.save('¹H₂.h5')

########
## Ar ##
########
t = levels.Atom(species='Ar')
t.load_from_nist('~/data/species/Ar/lines_levels/NIST_levels_2021-06-03.tsv')
t.save('Ar.h5')

## O
t = levels.Atom(species='O')
t.description = 'NIST database OI energy levels downloaded 2021-06-22.'
t.load_from_nist('~/data/species/O/lines_levels/NIST_levels_2021-06-22.tsv')
t.remove(np.isnan(t['E']))
t.save('O.h5')

## Kr
t = levels.Atom(species='Kr')
t.description = 'NIST database Kr I energy levels downloaded 2021-06-23.'
t.load_from_nist('~/data/species/Kr/lines_levels/NIST_levels_2021-06-23.tsv')
t.save('Kr.h5')

## Xe
t = levels.Atom(species='Xe')
t.description = 'NIST database Xe I energy levels downloaded 2021-07-30.'
t.load_from_nist('~/data/species/Xe/lines_levels/NIST_levels_2021-07-30.tsv')
t.save('Xe.h5')

########
## N2 ##
########
for species,fn in (
        ('¹⁴N₂','~/data/species/N2/lines_levels/ground_state/le_roy2006/14N2_relative_to_equilibrium'),
        ('¹⁴N¹⁵N','~/data/species/N2/lines_levels/ground_state/le_roy2006/14N15N_relative_to_equilibrium'),
        ('¹⁵N₂','~/data/species/N2/lines_levels/ground_state/le_roy2006/15N2_relative_to_equilibrium'),
        ):
    t = dataset.load(fn,labels_commented=True )
    d = levels.Diatom(species=species, label='X',
                        Ee=t['T'], J=t['J'], v=t['v'],
                        E0=np.min(t['T']), Eref=0,)
    d.save(f'{species}.h5')

########
## CO ##
########
##d = dataset.load('~/data/species/CO/ground_state/levels_coxon2004/all_levels.h5',classname='dataset.Dataset')
##d.pop('Fi')
##d['E0'] = d.pop('ZPE')
##print('DEBUG:', len(d))
##pprint(d.keys())
##x = levels.Diatom()
##for species in d.unique('species'):
##    print( species)
d = levels.Diatom()
d.load('~/data/species/CO/ground_state/levels_coxon2004/all_levels.h5',translate_keys={'ZPE':'E0',},)
for species in d.unique('species'):
    species = database.normalise_species(species)
    print( species)
    d.matches(species=species).save(f'{species}.h5')

