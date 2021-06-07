from spectr import *


# ########
# ## Xe ##
# ########
# t = lines.Atomic('Xe',load_from_string='''
# species | conf_u            | J_u | conf_l | J_l | ν            | L_l | S_l | gu_l | ν_unc   | reference
# Xe      | 5p5.6s            |     | 5p6    | 0   | 68045.171    | 0   | 0   | 1    | 0.005   | ozawa2013
# Xe      | 5p5.6s            |     | 5p6    | 0   | 83889.971    | 0   | 0   | 1    | 0.2     | yoshino1985
# Xe      | 5p5.6s            |     | 5p6    | 0   | 85439.92     | 0   | 0   | 1    | 0.2     | yoshino1985
# Xe      | 5p5.6s            |     | 5p6    | 0   | 90032.25     | 0   | 0   | 1    | 0.2     | yoshino1985
# Xe      | 5p5.8s            | 1.5 | 5p6    | 0   | 90932.44142  | 0   | 0   | 1    | 0.00002 | dreissen2019
# Xe      | 5p5.(2P*<3/2>).7d | 1.0 | 5p6    | 0.0 | 92128.450000 | 0   | 0   | 1    | nan     | NIST
# Xe      | 5p5.5d            | 1.5 | 5p6    | 0   | 93630        | 0   | 0   | 1    | nan     | 
# Xe      | 5p5.8d            | 0.5 | 5p6    | 0   | 94228.0817   | 0   | 0   | 1    | 0.0032  | brandi2001
# Xe      | 5p5.8d            | 1.5 | 5p6    | 0   | 94685.4822   | 0   | 0   | 1    | 0.0030  | brandi2001
# Xe      | 5p5.7s            | 0.5 | 5p6    | 0   | 95800.5867   | 0   | 0   | 1    | 0.0030  | brandi2001
# ''')
# t.save('Xe')
# t.save('Xe.h5')

# ########
# ## N ##
# ########
# t = lines.Atomic(species='N')
# t.load_from_nist('~/data/species/N/lines_levels/NIST_transitions_2020-06-29.psv')
# t.save('N.h5')

# ########
# ## Ar ##
# ########
# t = lines.Atomic(species='Ar')
# t.load_from_nist('~/data/species/Ar/lines_levels/NIST_transitions_20201-06-03.tsv')
# t.save('Ar.h5')

#######
## O ##
#######
# t = lines.Atomic(species='O')
# t.load_from_nist('~/data/species/O/NIST_transitions_2021-04-22.psv')
# t.save('O.h5')

# #######
# ## C ##
# #######
# t = lines.Atomic(species='C')
# t.load_from_nist('~/data/species/C/NIST_transitions_2021-04-22.psv')
# t.save('C.h5')

# # ########
# ## H2 ##
# ########
# d = lines.Diatomic(
    # description='H2 data downloaded from Meudon observatory "fichiers_all"',
    # species='H2',label_l='X')
# d.load('~/data/species/H2/lines_levels/meudon_observatory/fichiers_all',
       # labels_commented=True,
       # translate_keys={
           # 'state':'label_u',
           # 'vp':'v_u',
           # 'Jp':'J_u',
           # 'vpp':'v_l',
           # 'Jpp':'J_l',
           # 'A':'Ae',
           # 'nu':'ν',
           # 'At':'At',
           # 'Gamma':'Γ',
           # 'Ad':'Ad',
           # },
       # keys=('label_u', 'v_u', 'J_u', 'v_l', 'J_l', 'Ae', 'ν', 'Γ',
             # # 'At', 'Ad',
             # ),)
# d.limit_to_match(v_l_max=5,J_l_max=10)
# d.save('H2.h5')
