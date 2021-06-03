from spectr import *


########
## H2 ##
########
d = levels.Diatomic(species='H2',Tref=0)
d.load('~/data/species/H2/lines_levels/ground_state/H2_term_values_relative_to_fundamental_komasa2011',
       labels_commented=True,
       keys=('species','label','v','J','E'),
       translate_keys={'T':'E','dT':'E_unc',},)
d.save('H2.h5')
