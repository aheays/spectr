from spectr.env import *
# warnings.simplefilter('error')

tests = {

    '1Π-1Σ+_transition': {
         'levels':{
             'X.1Σ+(v=0)':{'Tv':0,'Bv':1},
             'B.1Π(v=0)':{'Tv':50000,'Bv':1.1}},
         'transition_moments':{
             ('B.1Π(v=0)','X.1Σ+(v=0)'):{'μv':1}}},

    '1Π_1Σ+_intensity_interference': {
        'levels':{
            'X.1Σ+(v=0)':{'Tv':0,'Bv':1}, 
            'B.1Π(v=0)':{'Tv':50000, 'Bv':1.1},
            'C.1Σ+(v=0)':{'Tv':49950, 'Bv':1.4}},
        'couplings':{
            ('B.1Π(v=0)','C.1Σ+(v=0)'):{'ξv':1}},
        'transition_moments':{
            ('B.1Π(v=0)','X.1Σ+(v=0)'):{'μv':1}, ('C.1Σ+(v=0)','X.1Σ+(v=0)'):{'μv':0.5}}},

    '2Σ+_2Π_transition': {
         'species':'¹⁴N¹⁶O',
         'levels':{
             'X.2Π(v=0)':{'Tv':100,'Bv':+1.6,'Av':+120,'γv':-0.01}, 
             'A.2Σ+(v=3)':{'Tv':50000,'Bv':1.1,'Dv':6e-6,'γv':10,'γDv':-1e-4}},
         'transition_moments':{
             ('A.2Σ+(v=3)','X.2Π(v=0)'):{'μv':1}}},

    '2Π_2Π_transition': {
         'species':'¹⁴N¹⁶O',
         'levels':{
             'X.2Π(v=0)':{'Tv':100,'Bv':+1.6,'Av':+120,'γv':-0.01},
             'B.2Π(v=0)':{'Tv':50000,'Av':100,'Bv':1.1,'Dv':6e-6,'γv':10,'γDv':-1e-4}}, 
         'transition_moments':{('B.2Π(v=0)','X.2Π(v=0)'):{'μv':1}}},

    '3Σ+-3Σ+_transition': {
         'levels':{
             'X.3Σ+(v=0)':{'Tv':0,'Bv':1,'λv':1,'γv':0.01},
             'C.3Σ+(v=0)':{'Tv':49950,'Bv':1.4,'Dv':1e-6,'λv':1,'λDv':1e-4,'γv':0.01,'γDv':1e-4},},
         'transition_moments':{
             ('C.3Σ+(v=0)','X.3Σ+(v=0)'):{'μv':1}}},

    '3Π-3Σ+_transition': {
         'levels':{
             'X.3Σ+(v=0)':{'Tv':0,'Bv':1,'λv':1,'γv':0.01},
             'B.3Π(v=0)':{'Tv':50000,'Bv':1.1,'Av':10}}, 
         'transition_moments':{
             ('B.3Π(v=0)','X.3Σ+(v=0)'):{'μv':1}}},

    '3Π-3Π_transition': {
         'levels':{
             'X.3Π(v=0)':{'Tv':0,'Bv':1,'Av':5,'λv':1,'γv':0.01},
             'B.3Π(v=0)':{'Tv':50000,'Bv':1.1,'Av':10}},
         'transition_moments':{
             ('B.3Π(v=0)','X.3Π(v=0)'):{'μv':1}}},

    ## # 14N2 W-B(3-1) from western201 -- not working  
    ## '3Δu_3Πg_transition': { 
    ##     'species':'¹⁴N₂',
    ##     'levels':{'B.3Πg(v=1)':{'Tv':11255.202248,'Bv':1.61055197,'Dv':5.89799e-06,'Hv':-3.94391e-13,'Av':42.1932,'ADv':-0.000421635,'λv':-0.203999,'λDv':-6.34857e-07,'γv':-0.00367735,'ov':1.15017,'pv':0.00431673,'qv':8.34856e-05,},
    ##               'W.3Δu(v=3)':{'Tv':13989.734287,'Bv':1.41063813,'Dv':5.65923e-06,'Hv':-6.54037e-13,'Av':5.6636,'ADv':3.92926e-06,'λv':0.67833,'λDv':1.51497e-06,'γv':-0.00288008,},},
    ##     'transition_moments':{('W.3Δu(v=3)','B.3Πg(v=1)'):{'μv':1}}},

    '3Π_3Σ+_intensity_interference': {
        'species':'³²S¹⁶O',
        'levels':{
            'X.3Σ+(v=0)':{'Tv':0, 'Bv':1, 'λv':1, 'γv':0.01},
            'B.3Π(v=0)': {'Tv':50000, 'Bv':1.1, 'Av':10,}, 'C.3Σ+(v=0)':{'Tv':49950, 'Bv':1.4, 'λv':1, 'γv':0.01,},},
        'couplings':{
            ('B.3Π(v=0)','C.3Σ+(v=0)'): {'ξv':0.1, 'ηv':10},},
        'transition_moments':{
            ('B.3Π(v=0)','X.3Σ+(v=0)'):{'μv':1},
            ('C.3Σ+(v=0)','X.3Σ+(v=0)'):{'μv':1}}},

    '3Π-3Σ-_transition': {
         'species':'³²S¹⁶O',
         'levels':{
             'X.3Σ-(v=0)':{'Tv':0, 'Bv':convert.units(21523.555878,'MHz','cm-1'), 'Dv':convert.units(33.915261e-3,'MHz','cm-1'), 'Hv':convert.units(-6.974e-9,'MHz','cm-1'), 'λv':convert.units(158254.392,'MHz','cm-1'), 'λDv':convert.units(0.306259,'MHz','cm-1'), 'λHv':convert.units(0.478e-6,'MHz','cm-1'), 'γv':convert.units(-168.3043,'MHz','cm-1'), 'γDv':convert.units(-0.52545e-3,'MHz','cm-1'),},
             'C.3Π(v=0)':{'Tv':44151.8,'Bv':0.567,'Dv':1.2e-5,'Av':-181.4,'λv':1.0,'ov':0.98,'γv':-0.2,}},
         'transition_moments':{
             ('C.3Π(v=0)','X.3Σ-(v=0)'):{'μv':0.5}}},

    '32S16O_B05_C00_d01': {
        'species':'³²S¹⁶O',
        'levels':{
            'X.3Σ-(v=0)':{'Tv':0,'Bv':convert.units(21523.555878,'MHz','cm-1'),'Dv':convert.units(33.915261e-3,'MHz','cm-1'),'Hv':convert.units(-6.974e-9,'MHz','cm-1'),'λv':convert.units(158254.392,'MHz','cm-1'),'λDv':convert.units(0.306259,'MHz','cm-1'),'λHv':convert.units(0.478e-6,'MHz','cm-1'),'γv':convert.units(-168.3043,'MHz','cm-1'),'γDv':convert.units(-0.52545e-3,'MHz','cm-1')},
            'B.3Σ-(v=5)':{'Tv':44382.05,'Bv':0.4705,'Dv':-0.8e-6,'λv':1.0,'γv':-1.9e-2,},
            'C.3Π(v=0)':{'Tv':44151.8,'Bv':0.567,'Dv':1.2e-5,'Av':-181.4,'λv':1.0,'ov':0.98,'γv':-0.2,},
            'd.1Π(v=1)':{'Tv':44143.2,'Bv':0.626}},
            'couplings':{
                ('C.3Π(v=0)','B.3Σ-(v=5)'):{'ξv':0.32,'ξDv':-4.1e-4},
                ('C.3Π(v=0)','B.3Σ-(v=5)'):{'ηv':2.4,'ηDv':0.015} ,
                ('C.3Π(v=0)','B.3Σ-(v=5)'):{'ξv':0.32,'ξDv':-4.1e-4,'ηv':2.4,'ηDv':0.015},
                ('C.3Π(v=0)','d.1Π(v=1)' ):{'ηv':8.7},} ,
         'transition_moments':{
             ('B.3Σ-(v=5)','X.3Σ-(v=0)'):{'μv':1},
             ('C.3Π(v=0)','X.3Σ-(v=0)'):{'μv':0.5}}},

    ## coupling pv not implemented
    ## '32S16O_B01_A10_App14_simplified': { # from liu_ching-ping2006, leave out <B|LS|App>_because of phase proble: 
    ##      'species':'³²S¹⁶O',
    ##     'levels':{
    ##         'X.3Σ-(v=0)':{ 'Tv':573.79105, 'Bv':0.71794874, 'Dv':1.1313242e-06, 'Hv':-2.1611497e-13, 'λv':5.2784506, 'λDv':1.0213915e-05, 'λHv':1.6510442e-11, 'γv':-0.0056139085, 'γDv':-1.7618215e-08,},
    ##         'B.3Σ-(v=1)':{'Tv':41991.97, 'Bv':0.49426 , 'Dv':1.61e-6 , 'λv':3.20    , 'γv':-1.37e-2,},
    ##         'A.3Π(v=10)':{'Tv':41990.00, 'Bv':0.4618  , 'Av':134.86  , 'λv':1.67    , 'ov':0.45    , 'Dv':4e-7    ,},
    ##         'App.3Σ+(v=14)':{'Tv':42007.35  , 'Bv':0.256     , 'λv':-2.66     , 'γv':0.81      ,}},
    ##     'couplings':{
    ##         ('B.3Σ-(v=1)','App.3Σ+(v=14)'):{'pv':1.87},
    ##         ('B.3Σ-(v=1)','A.3Π(v=10)'   ):{'ξv':-0.020},},
    ##     'transition_moments':{
    ##         ('B.3Σ-(v=1)','X.3Σ-(v=0)'):{'μv':1}}},

    ## coupling pv not implemented
    ##  '32S16O_B01_A10_App14': { # from liu_ching-ping2006 -- currently bad phases!!:
    ##      'species':'³²S¹⁶O',
    ##      'levels':{
    ##          'X.3Σ-(v=0)':{'Tv':573.79105, 'Bv':0.71794874, 'Dv':1.1313242e-06, 'Hv':-2.1611497e-13, 'λv':5.2784506, 'λDv':1.0213915e-05, 'λHv':1.6510442e-11, 'γv':-0.0056139085, 'γDv':-1.7618215e-08,},
    ##          'B.3Σ-(v=1)':{'Tv':41991.97,'Bv':0.49426,'Dv':1.61e-6,'λv':3.20,'γv':-1.37e-2},
    ##          'A.3Π(v=10)':{'Tv':41990.00,'Bv':0.4618,'Av':134.86,'λv':1.67,'ov':0.45,'Dv':4e-7},
    ##          'App.3Σ+(v=14)':{'Tv':42007.35,'Bv':0.256,'λv':-2.66,'γv':0.81,}},
    ##      'couplings': {
    ##          ('B.3Σ-(v=1)','A.3Π(v=10)'   ):{'ξv':-0.020},
    ##          ('B.3Σ-(v=1)','App.3Σ+(v=14)'):{'pv':-0.161,'ηv':1.87},},
    ##      'transition_moments':{
    ##          ('B.3Σ-(v=1)','X.3Σ-(v=0)'):{'μv':1}}},

    ## coupling pv not implemented
    ## '32S16O_B01_App14':{ # test phases of <B|Sun|App> and <B|LS|App> -- they do not agree. Change the sign fo one to correctly agree with the other half of the line
    ##      'species':'³²S¹⁶O',
    ##     'levels':{
    ##         'X.3Σ-(v=0)':{'Tv':573.79105, 'Bv':0.71794874, 'Dv':1.1313242e-06, 'Hv':-2.1611497e-13, 'λv':5.2784506, 'λDv':1.0213915e-05, 'λHv':1.6510442e-11, 'γv':-0.0056139085, 'γDv':-1.7618215e-08,},
    ##         'B.3Σ-(v=1)':{ 'Tv':41991.97, 'Bv':0.49426 , 'Dv':1.61e-6 , 'λv':3.20    , 'γv':-1.37e-2,},
    ##         'A.3Π(v=10)':{'Tv':41990.00, 'Bv':0.4618  , 'Av':134.86  , 'λv':1.67    , 'ov':0.45    , 'Dv':4e-7    ,},
    ##         'App.3Σ+(v=14)':{ 'Tv':42007.35  , 'Bv':0.256     , 'λv':-2.66     , 'γv':0.81      ,}},
    ##     'couplings':{
    ##         ('B.3Σ-(v=1)','App.3Σ+(v=14)'):{'pv':1.87,'ηv':1}},
    ##     'transition_moments':{
    ##         ('B.3Σ-(v=1)','X.3Σ-(v=0)'):{'μv':1}}},

}


for itest,(name,params) in enumerate(tests.items()):

    print( name)

    params.setdefault('species','³²S¹⁶O')
    # # params.setdefault('J_l',range(21))
    # params.setdefault('J_l',arange(1.5,2.5))
    params.setdefault('ΔJ',(-1,0,1))

    ## load model
    line_mod = viblevel.calc_line(**params)

    ## load pgopher
    line_pgo = lines.Diatom(species=params['species'])
    line_pgo.load_from_pgopher(f"data/{name}.csv",)

    ## limit to common levels
    max_J_l = min(np.max(line_pgo['J_l']),np.max(line_mod['J_l']))-1
    line_mod.limit_to_match(max_J_l=max_J_l)
    line_pgo.limit_to_match(max_J_l=max_J_l)

    ## needed to calculate ground state population
    line_pgo['Teq'] = line_mod['Teq'] = 300
    line_pgo['Zsource'] = line_mod['Zsource'] = 'unity'
    line_pgo['ΓD'] = line_mod['ΓD'] = 0.1
    line_pgo['Zsource'] = line_mod['Zsource'] = 'self'
    line_pgo['species'] = line_mod['species'][0]

    ## plot comparison of spectra
    # ykey = 'σ'
    ykey = 'Sij'

    xpgo,ypgo = line_pgo.calculate_spectrum(xkey='ν',ykey=ykey,dx=0.01,lineshape='gaussian')
    xmod,ymod = line_mod.calculate_spectrum(xkey='ν',ykey=ykey,x=xpgo,lineshape='gaussian')

    ## plot spectra
    qfig(itest)
    title(name)
    plot(xmod,ymod,label=f'mod, int(mod) = {integrate(ymod,xmod):0.5e}')
    plot(xpgo,ypgo,label=f'pgo, int(pgo) = {integrate(ypgo,xpgo):0.5e}')
    fractional_integrated_error = abs(integrate(ymod-ypgo,xpgo)/integrate(ypgo,xpgo))
    plot(xmod,(ymod-ypgo),label=f'mod-pgo, int(mod-pgo)/int(pgo) = {fractional_integrated_error:0.5e}')
    legend_colored_text(loc='upper left')
    assert fractional_integrated_error < 1e-5, f'test model: {name}: Integrated residual error {fractional_integrated_error} between viblevel model and pgopher is too large.'

show()


