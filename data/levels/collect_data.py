from spectr.env import *

## H2
d = levels.Diatom(name='H2_levels',species='¹H₂')
d.load(
    '~/data/species/H2/lines_levels/ground_state/H2_term_values_relative_to_fundamental_komasa2011',
    labels_commented=True,
    filetype='simple_text',
    keys=('species','label','v','J','E'),
    translate_keys={'T':'E','dT':'E_unc',},)
#  d['species'] = '¹H₂'

d['Eref'] = 0
d.save('¹H₂.h5')
d.describe()

## Ar
t = levels.Atom(name='Ar_levels')
t.load_from_nist('~/data/species/Ar/lines_levels/NIST_levels_2021-06-03.tsv')
t['species'] = 'Ar'
t.save('Ar.h5')
t.describe()

## O
t = levels.Atom(name='O_levels')
t.description = 'NIST database OI energy levels downloaded 2021-06-22.'
t.load_from_nist('~/data/species/O/lines_levels/NIST_levels_2021-06-22.tsv')
t.remove(np.isnan(t['E']))
t['species'] = 'O'
t.save('O.h5')
t.describe()

## Kr
t = levels.Atom(name='Kr_levels')
t.description = 'NIST database Kr I energy levels downloaded 2021-06-23.'
t.load_from_nist('~/data/species/Kr/lines_levels/NIST_levels_2021-06-23.tsv')
t['species'] = 'Kr'
t.save('Kr.h5')
t.describe()

## Xe
t = levels.Atom(name='Xe_levels')
t.description = 'NIST database Xe I energy levels downloaded 2021-07-30.'
t.load_from_nist('~/data/species/Xe/lines_levels/NIST_levels_2021-07-30.tsv')
t['species'] = 'Xe'
t.save('Xe.h5')
t.describe()

## N2
for species,fn in (
        ('¹⁴N₂','~/data/species/N2/lines_levels/ground_state/le_roy2006/14N2_relative_to_equilibrium'),
        ('¹⁴N¹⁵N','~/data/species/N2/lines_levels/ground_state/le_roy2006/14N15N_relative_to_equilibrium'),
        ('¹⁵N₂','~/data/species/N2/lines_levels/ground_state/le_roy2006/15N2_relative_to_equilibrium'),
        ):
    t = dataset.load(fn,labels_commented=True,filetype='simple_text')
    d = levels.Diatom(
        name='N2_levels',
        species=species, label='X',
        Ee=t['T'], J=t['J'], v=t['v'],
        E0=np.min(t['T']), Eref=0,)
    d.save(f'{species}.h5')
d.describe()
    
## CO
d = levels.Diatom(name='CO_levels')
d.load('~/data/species/CO/ground_state/levels_coxon2004/all_levels.h5',translate_keys={'ZPE':'E0','classname':None},)
for species in d.unique('species'):
    d['species',d.match(species=species)] = database.normalise_species(species)

for species in d.unique('species'):
    t = d.matches(species=species)
    t.describe()
    t.save(f'{species}.h5')

