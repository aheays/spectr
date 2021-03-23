from spectr import *

tests = [

    # dict(name='1Σ+-1Σ+_transition',
         # lower_levels=({'name':'X.1Σ+(v=0)','Tv':0,'Bv':1},), 
         # upper_levels=({'name':'B.1Σ+(v=0)','Tv':50000,'Bv':1.1},), 
         # transition_moments=({'name_u':'B.1Σ+(v=0)','name_l':'X.1Σ+(v=0)','μv':1},),),

    # dict(name='1Π-1Σ+_transition',
         # lower_levels=({'name':'X.1Σ+(v=0)','Tv':0,'Bv':1},), 
         # upper_levels=({'name':'B.1Π(v=0)','Tv':50000,'Bv':1.1},), 
         # transition_moments=({'name_u':'B.1Π(v=0)','name_l':'X.1Σ+(v=0)','μv':1},),),

    # dict(name='1Π_1Σ+_intensity_interference',
         # lower_levels=({'name':'X.1Σ+(v=0)','Tv':0,'Bv':1},), 
         # upper_levels=({'name':'B.1Π(v=0)','Tv':50000,'Bv':1.1},
                       # {'name':'C.1Σ+(v=0)','Tv':49950,'Bv':1.4},), 
         # upper_coupling=({'type':'JL','name1':'B.1Π(v=0)','name2':'C.1Σ+(v=0)','ξv':1},),
         # transition_moments=({'name_u':'B.1Π(v=0)','name_l':'X.1Σ+(v=0)','μv':1},  
                             # {'name_u':'C.1Σ+(v=0)','name_l':'X.1Σ+(v=0)','μv':0.5},),),

    # dict(name='2Σ+_2Π_transition',
         # species='[14N][16O]',
         # J=np.arange(0.5,31),
         # lower_levels=({'name':'X.2Π(v=0)','Tv':100,'Bv':+1.6,'Av':+120,'γv':-0.01},), 
         # upper_levels=({'name':'A.2Σ+(v=3)','Tv':50000,'Bv':1.1,'Dv':6e-6,'γv':10,'γDv':-1e-4},), 
         # transition_moments=({'name_u':'A.2Σ+(v=3)','name_l':'X.2Π(v=0)','μv':1},),),

    # dict(name='2Π_2Π_transition',
         # species='[14N][16O]',
         # J=np.arange(0.5,31),
         # lower_levels=({'name':'X.2Π(v=0)','Tv':100,'Bv':+1.6,'Av':+120,'γv':-0.01},), 
         # upper_levels=({'name':'B.2Π(v=0)','Tv':50000,'Av':100,'Bv':1.1,'Dv':6e-6,'γv':10,'γDv':-1e-4},), 
         # transition_moments=({'name_u':'B.2Π(v=0)','name_l':'X.2Π(v=0)','μv':1},),),

    # dict(name='3Σ+-3Σ+_transition',
         # lower_levels=({'name':'X.3Σ+(v=0)','Tv':0,'Bv':1,'λv':1,'γv':0.01},), 
         # upper_levels=({'name':'C.3Σ+(v=0)','Tv':49950,'Bv':1.4,'Dv':1e-6,'λv':1,'λDv':1e-4,'γv':0.01,'γDv':1e-4},), 
         # transition_moments=({'name_u':'C.3Σ+(v=0)','name_l':'X.3Σ+(v=0)','μv':1},),),

    # dict(name='3Π-3Σ+_transition',
         # lower_levels=({'name':'X.3Σ+(v=0)','Tv':0,'Bv':1,'λv':1,'γv':0.01},), 
         # upper_levels=({'name':'B.3Π(v=0)','Tv':50000,'Bv':1.1,'Av':10},), 
         # transition_moments=({'name_u':'B.3Π(v=0)','name_l':'X.3Σ+(v=0)','μv':1},),),

    # dict(name='3Π-3Π_transition',
         # lower_levels=({'name':'X.3Π(v=0)','Tv':0,'Bv':1,'Av':5,'λv':1,'γv':0.01},), 
         # upper_levels=({'name':'B.3Π(v=0)','Tv':50000,'Bv':1.1,'Av':10},), 
         # transition_moments=({'name_u':'B.3Π(v=0)','name_l':'X.3Π(v=0)','μv':1},),),

     dict(name='3Δu_3Πg_transition', # 14N2 W-B(3-1) from western2018
        species='[14N]2',
          lower_levels=({'name':'B.3Πg(v=1)','Tv':11255.202248,'Bv':1.61055197,'Dv':5.89799e-06,'Hv':-3.94391e-13,'Av':42.1932,'ADv':-0.000421635,'λv':-0.203999,'λDv':-6.34857e-07,'γv':-0.00367735,'ov':1.15017,'pv':0.00431673,'qv':8.34856e-05,},),
        upper_levels=({'name':'W.3Δu(v=3)','Tv':13989.734287,'Bv':1.41063813,'Dv':5.65923e-06,'Hv':-6.54037e-13,'Av':5.6636,'ADv':3.92926e-06,'λv':0.67833,'λDv':1.51497e-06,'γv':-0.00288008,},),
        transition_moments=({'name_u':'W.3Δu(v=3)','name_l':'B.3Πg(v=1)','μv':1},),),

    # dict(name='3Π_3Σ+_intensity_interference',
         # lower_levels=({'name':'X.3Σ+(v=0)','Tv':0,'Bv':1,'λv':1,'γv':0.01},), 
         # upper_levels=({'name':'B.3Π(v=0)','Tv':50000,'Bv':1.1,'Av':10},
                       # {'name':'C.3Σ+(v=0)','Tv':49950,'Bv':1.4,'λv':1,'γv':0.01,}), 
         # upper_coupling=(
             # {'type':'JL','name1':'B.3Π(v=0)','name2':'C.3Σ+(v=0)','ξv':0.1},
             # {'type':'LS','name1':'B.3Π(v=0)','name2':'C.3Σ+(v=0)','ηv':10},
         # ),
         # transition_moments=({'name_u':'B.3Π(v=0)','name_l':'X.3Σ+(v=0)','μv':1},
                             # {'name_u':'C.3Σ+(v=0)','name_l':'X.3Σ+(v=0)','μv':1}, ),),

    # dict(name='3Π-3Σ-_transition',
         # species='[32S][16O]',
         # lower_levels=({'name':'X.3Σ-(v=0)',
                        # 'Tv':0,
                        # 'Bv':convert.units(21523.555878,'MHz','cm-1'),
                        # 'Dv':convert.units(33.915261e-3,'MHz','cm-1'),
                        # 'Hv':convert.units(-6.974e-9,'MHz','cm-1'),
                        # 'λv':convert.units(158254.392,'MHz','cm-1'),
                        # 'λDv':convert.units(0.306259,'MHz','cm-1'),
                        # 'λHv':convert.units(0.478e-6,'MHz','cm-1'),
                        # 'γv':convert.units(-168.3043,'MHz','cm-1'),
                        # 'γDv':convert.units(-0.52545e-3,'MHz','cm-1'),
                        # },),
         # upper_levels=({'name':'C.3Π(v=0)','Tv':44151.8,'Bv':0.567,'Dv':1.2e-5,'Av':-181.4,'λv':1.0,'ov':0.98,'γv':-0.2,},),
         # transition_moments=({'name_u':'C.3Π(v=0)','name_l':'X.3Σ-(v=0)','μv':0.5},),),

    # dict(name='32S16O_B05_C00_d01',
         # species='[32S][16O]',
         # lower_levels=({'name':'X.3Σ-(v=0)','Tv':0,'Bv':convert.units(21523.555878,'MHz','cm-1'),'Dv':convert.units(33.915261e-3,'MHz','cm-1'),'Hv':convert.units(-6.974e-9,'MHz','cm-1'),'λv':convert.units(158254.392,'MHz','cm-1'),'λDv':convert.units(0.306259,'MHz','cm-1'),'λHv':convert.units(0.478e-6,'MHz','cm-1'),'γv':convert.units(-168.3043,'MHz','cm-1'),'γDv':convert.units(-0.52545e-3,'MHz','cm-1'),},),
         # upper_levels=({'name':'B.3Σ-(v=5)','Tv':44382.05,'Bv':0.4705,'Dv':-0.8e-6,'λv':1.0,'γv':-1.9e-2,},
                       # {'name':'C.3Π(v=0)','Tv':44151.8,'Bv':0.567,'Dv':1.2e-5,'Av':-181.4,'λv':1.0,'ov':0.98,'γv':-0.2,},
                       # {'name':'d.1Π(v=1)','Tv':44143.2,'Bv':0.626},),
         # upper_coupling=(
             # {'type':'JL','name1':'C.3Π(v=0)','name2':'B.3Σ-(v=5)','ξv':0.32,'ξDv':-4.1e-4},
             # {'type':'LS','name1':'C.3Π(v=0)','name2':'B.3Σ-(v=5)','ηv':2.4,'ηDv':0.015},
             # {'type':'LS','name1':'C.3Π(v=0)','name2':'d.1Π(v=1)','ηv':8.7},
         # ),
         # transition_moments=({'name_u':'B.3Σ-(v=5)','name_l':'X.3Σ-(v=0)','μv':1},
                             # {'name_u':'C.3Π(v=0)','name_l':'X.3Σ-(v=0)','μv':0.5},),),

    # dict(name='32S16O_B01_A10_App14_simplified', # from liu_ching-ping2006, leave out <B|LS|App>_because of phase problem
         # species='[32S][16O]',
         # lower_levels=({'name':'X.3Σ-(v=0)', 'Tv':573.79105, 'Bv':0.71794874, 'Dv':1.1313242e-06, 'Hv':-2.1611497e-13, 'λv':5.2784506, 'λDv':1.0213915e-05, 'λHv':1.6510442e-11, 'γv':-0.0056139085, 'γDv':-1.7618215e-08,},),
         # upper_levels=(
             # {'name':'B.3Σ-(v=1)', 'Tv':41991.97, 'Bv':0.49426 , 'Dv':1.61e-6 , 'λv':3.20    , 'γv':-1.37e-2,},
             # {'name':'A.3Π(v=10)', 'Tv':41990.00, 'Bv':0.4618  , 'Av':134.86  , 'λv':1.67    , 'ov':0.45    , 'Dv':4e-7    ,},
             # {'name':'App.3Σ+(v=14)', 'Tv':42007.35  , 'Bv':0.256     , 'λv':-2.66     , 'γv':0.81      ,},
         # ),
         # upper_coupling=(
             # {'type':'JS','name1':'B.3Σ-(v=1)','name2':'App.3Σ+(v=14)','pv':1.87},
             # {'type':'JL','name1':'B.3Σ-(v=1)','name2':'A.3Π(v=10)','ξv':-0.020},
         # ),
         # transition_moments=({'name_u':'B.3Σ-(v=1)','name_l':'X.3Σ-(v=0)','μv':1},),),

     # dict(name='32S16O_B01_A10_App14', # from liu_ching-ping2006 -- currently bad phases!!!
         # species='[32S][16O]',
         # lower_levels=({'name':'X.3Σ-(v=0)', 'Tv':573.79105, 'Bv':0.71794874, 'Dv':1.1313242e-06, 'Hv':-2.1611497e-13, 'λv':5.2784506, 'λDv':1.0213915e-05, 'λHv':1.6510442e-11, 'γv':-0.0056139085, 'γDv':-1.7618215e-08,},),
         # upper_levels=({'name':'B.3Σ-(v=1)', 'Tv':41991.97, 'Bv':0.49426 , 'Dv':1.61e-6 , 'λv':3.20    , 'γv':-1.37e-2,},
                       # {'name':'A.3Π(v=10)', 'Tv':41990.00, 'Bv':0.4618  , 'Av':134.86  , 'λv':1.67    , 'ov':0.45    , 'Dv':4e-7    ,},
                       # {'name':'App.3Σ+(v=14)', 'Tv':42007.35  , 'Bv':0.256     , 'λv':-2.66     , 'γv':0.81      ,},),
         # upper_coupling=(
             # {'type':'JS','name1':'B.3Σ-(v=1)','name2':'App.3Σ+(v=14)','pv':-0.161},
             # {'type':'JL','name1':'B.3Σ-(v=1)','name2':'A.3Π(v=10)','ξv':-0.020},
             # {'type':'LS','name1':'B.3Σ-(v=1)','name2':'App.3Σ+(v=14)','ηv':1.87},
         # ),
         # transition_moments=({'name_u':'B.3Σ-(v=1)','name_l':'X.3Σ-(v=0)','μv':1},),),



    # dict(name='32S16O_B01_App14', # test phases of <B|Sun|App> and <B|LS|App> -- they do not agree. Change the sign fo one to correctly agree with the other half of the lines
         # species='[32S][16O]',
         # lower_levels=({'name':'X.3Σ-(v=0)', 'Tv':573.79105, 'Bv':0.71794874, 'Dv':1.1313242e-06, 'Hv':-2.1611497e-13, 'λv':5.2784506, 'λDv':1.0213915e-05, 'λHv':1.6510442e-11, 'γv':-0.0056139085, 'γDv':-1.7618215e-08,},),
         # upper_levels=(
             # {'name':'B.3Σ-(v=1)', 'Tv':41991.97, 'Bv':0.49426 , 'Dv':1.61e-6 , 'λv':3.20    , 'γv':-1.37e-2,},
             # {'name':'A.3Π(v=10)', 'Tv':41990.00, 'Bv':0.4618  , 'Av':134.86  , 'λv':1.67    , 'ov':0.45    , 'Dv':4e-7    ,},
             # {'name':'App.3Σ+(v=14)', 'Tv':42007.35  , 'Bv':0.256     , 'λv':-2.66     , 'γv':0.81      ,},
         # ),
         # upper_coupling=(
             # {'type':'JS','name1':'B.3Σ-(v=1)','name2':'App.3Σ+(v=14)','pv':1.87},
             # {'type':'LS','name1':'B.3Σ-(v=1)','name2':'App.3Σ+(v=14)','ηv':1},
         # ),
         # transition_moments=({'name_u':'B.3Σ-(v=1)','name_l':'X.3Σ-(v=0)','μv':1},),),


]

for itest,test in enumerate(tests):
    test.setdefault('species','[32S][16O]')
    test.setdefault('J',range(21))
    # test.setdefault('J',[5])
    test.setdefault('Teq',300.)

    ## lower state
    lower = viblevel.VibLevel(name='lower',species=test['species'])
    for kwargs in test['lower_levels']:
        lower.add_level(**kwargs)

    ## upper state
    upper = viblevel.VibLevel(name='upper',species=test['species'])
    for kwargs in test['upper_levels']:
        upper.add_level(**kwargs)

    if 'upper_coupling' in test:
        for kwargs in test['upper_coupling']:
            coupling_type = kwargs.pop('type')
            if coupling_type == 'JL':
                upper.add_JL_coupling(**kwargs)
            elif coupling_type == 'LS':
                upper.add_LS_coupling(**kwargs)
            elif coupling_type == 'JS':
                upper.add_JS_coupling(**kwargs)
            else:
                raise Exception

    # if 'upper_add_L_uncoupling' in test:
        # for kwargs in test['upper_add_L_uncoupling']: 
            # upper.add_L_uncoupling(**kwargs)
    # if 'upper_add_S_uncoupling' in test:
        # for kwargs in test['upper_add_S_uncoupling']:
            # upper.add_S_uncoupling(**kwargs)
    # if 'upper_add_LS_coupling' in test:
        # for kwargs in test['upper_add_LS_coupling']: upper.add_LS_coupling(**kwargs)

    ## transition
    transition = viblevel.VibLine('transition',upper,lower,J_l=test['J'],)
    for kwargs in test['transition_moments']:
        transition.add_transition_moment(**kwargs)
    transition.construct()
    line_mod = transition.rotational_line

    ## pgopher
    line_pgo = lines.Generic(species=test['species'])
    line_pgo.load_from_pgopher(f"data/{test['name']}.csv",)

    ## limit to common levels
    J_l_max = min(np.max(line_pgo['J_l']),np.max(line_mod['J_l']))
    line_mod.limit_to_match(J_l_max=J_l_max)
    line_pgo.limit_to_match(J_l_max=J_l_max)

    ## needed to calculate ground state population
    line_pgo['Teq'] = line_mod['Teq'] = test['Teq']
    line_pgo['ΓD'] = line_mod['ΓD'] = 0.1
    line_pgo['Zsource'] = line_mod['Zsource'] = 'self'

    ## plot comparison of spectra
    # ykey = 'σ'
    ykey = 'Sij'

    xpgo,ypgo = line_pgo.calculate_spectrum(xkey='ν',ykey=ykey,dx=0.01)
    xmod,ymod = line_mod.calculate_spectrum(xkey='ν',ykey=ykey,x=xpgo)

    ## plot spectra
    qfig(itest)
    title(test['name'])
    plot(xmod,ymod,label=f'mod, int(mod) = {integrate.trapz(ymod,xmod):0.5e}')
    plot(xpgo,ypgo,label=f'pgo, int(pgo) = {integrate.trapz(ypgo,xpgo):0.5e}')
    plot(xmod,(ymod-ypgo),label=f'mod-pgo, int(mod-pgo)/int(pgo) = {integrate.trapz(ymod-ypgo,xpgo)/integrate.trapz(ypgo,xpgo):0.5e}')
    legend_colored_text(loc='upper left')

    # print( upper.rotational_level.matches(J=5))

show()


